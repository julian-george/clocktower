"""Beaker integration: hillclimb the harness of one open-weight Clocktower player (clocktower/player/).

A case is one deal (seed) and one good focal seat, played for two days. The focal seat uses the open-weight
model and the player harness Beaker edits; the other seats are fixed closed models. The score is how
accurately the focal seat's belief probe names the evil players and the Demon.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from pydantic import BaseModel, Field

from beaker import (
    Case, CaseResult, CaseScore, Check, Integration,
    RetryableCaseError, RolloutRuntime, RunSetup, SetupRuntime, repository,
)

FOCAL_MODEL = "openrouter:z-ai/glm-5.3-flash"
OPPONENT_MODELS = "claude-sonnet-5-5,gpt-6.1-terra,gpt-6-luna"
CHECKOUT_ROOT = Path(__file__).resolve().parent.parent
PLAYER_HARNESS_FILES = ("clocktower/player/playbook.md", "clocktower/player/scaffold.py")
PROVIDER_KEY_NAMES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")
MAX_SPEND_PER_GAME_USD = 25
GAME_DAYS = 2  # a short game: enough talk to form beliefs, a fraction of a full game
PASSING_ACCURACY = 0.85
MAX_TRANSCRIPT_CHARS_IN_TRACE = 100_000


class Row(BaseModel):
    id: str = Field(min_length=1)
    seed: int
    focal_seat: int = Field(ge=0, lt=10)  # tables seat at least 10 players


class Setup(RunSetup[Row]):
    row_model = Row

    async def load_cases(self, row: Row, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        yield Case(id=row.id, input={"seed": row.seed, "focal_seat": row.focal_seat},
                   expected={"metric": "belief accuracy about evil players and the Demon"})


def run_clocktower_cli(arguments: list[str]) -> None:
    """Runs one game on this thread's own event loop; to_thread keeps the Beaker trace context."""
    if str(CHECKOUT_ROOT) not in sys.path:
        sys.path.insert(0, str(CHECKOUT_ROOT))
    from clocktower.cli import main
    try:
        main(arguments)
    except SystemExit as exit_request:
        if exit_request.code not in (0, None):
            raise RuntimeError(f"clocktower exited with {exit_request.code}") from exit_request


def belief_accuracy(probe: dict) -> float:
    """1 minus the mean squared error of the seat's evil and Demon odds for every other player (unanswered counts as 50%)."""
    from clocktower.characters import DEMONS
    errors = []
    for name, truth in probe["truth"].items():
        if name == probe["player"]:
            continue
        answered = probe.get("players", {}).get(name) or {}
        for key, actual in (("evil", truth["evil"]), ("demon", truth["character"] in DEMONS)):
            predicted = answered.get(key)
            errors.append(((0.5 if predicted is None else predicted) - float(actual)) ** 2)
    return 1 - sum(errors) / len(errors)


async def play_game(seed: int, focal_seat: int, game_dir: Path, trace) -> dict:
    # In-process (not a subprocess) so the model calls in clocktower/llm.py join Beaker's trace capture.
    with trace.stage("game", inputs={"seed": seed, "focal_seat": focal_seat, "focal_model": FOCAL_MODEL,
                                     "opponent_models": OPPONENT_MODELS, "days": GAME_DAYS}):
        try:
            await asyncio.to_thread(run_clocktower_cli, [
                "run", "--model", OPPONENT_MODELS, "--focal-seat", str(focal_seat), "--focal-model", FOCAL_MODEL,
                "--seed", str(seed), "--days", str(GAME_DAYS), "--probe-focal-only",
                "--budget", str(MAX_SPEND_PER_GAME_USD), "--out", str(game_dir),
            ])
        except RuntimeError as game_failure:
            raise RetryableCaseError(f"game failed: {game_failure}") from game_failure
        result = json.loads((game_dir / "result.json").read_text())
        probe_lines = (game_dir / "probes.jsonl").read_text().splitlines() if (game_dir / "probes.jsonl").exists() else []
        probes = [json.loads(line) for line in probe_lines]
        transcript = game_dir / "transcript.md"
        if transcript.exists():
            trace.evidence("transcript", transcript.read_text()[:MAX_TRANSCRIPT_CHARS_IN_TRACE])
        trace.evidence("probes", probes)
    answered = [probe for probe in probes if probe.get("players")]
    if not answered:
        raise RetryableCaseError(f"no belief probe was answered: {[probe.get('error') for probe in probes]}")
    last = max(answered, key=lambda probe: (probe["day"], probe["point"] == "nominations"))
    return {"focal_seat": focal_seat, "focal_name": last["player"], "focal_character": last["character"],
            "belief_accuracy": belief_accuracy(last), "probe_day": last["day"], "probe_point": last["point"],
            "winner": result.get("winner"), "cost_usd": result["cost"]}


async def run_case(*, case_input: dict, runtime: RolloutRuntime) -> CaseResult:
    for harness_file in PLAYER_HARNESS_FILES:
        if not (CHECKOUT_ROOT / harness_file).exists():
            raise RuntimeError(f"{harness_file} missing under {CHECKOUT_ROOT}")
    # The harness insists on keys; with none set, hosted provider routing supplies platform credentials.
    for key_name in PROVIDER_KEY_NAMES:
        os.environ.setdefault(key_name, "beaker-provider-routing")
    with tempfile.TemporaryDirectory(prefix="clocktower-game-") as temp_dir:
        game = await play_game(case_input["seed"], case_input["focal_seat"], Path(temp_dir), runtime.trace)
    return CaseResult(output=game, output_kind="record")


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    game = result.output
    accuracy = game["belief_accuracy"]
    check = Check(
        name="beliefs about who is evil and the Demon", verdict="pass" if accuracy >= PASSING_ACCURACY else "fail",
        expected=f">= {PASSING_ACCURACY}", predicted=f"{accuracy:.3f}",
        message=f"{game['focal_name']} ({game['focal_character']}) after day {game['probe_day']} {game['probe_point']}",
    )
    return CaseScore(objective=accuracy, field_scores={"belief_accuracy": accuracy}, checks=(check,))


integration = Integration(
    targets=repository(PLAYER_HARNESS_FILES),
    run_setup=Setup, run_case=run_case, score_case=score_case,
)
