"""Beaker integration: hillclimb the harness of one open-weight Clocktower player (clocktower/player/).

A case is one deal (seed) and one focal seat. The focal seat plays with the open-weight model and the
player harness Beaker edits; every other seat is a fixed mid-priced closed model. The score is whether
the focal seat's team wins.
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
MAX_TRANSCRIPT_CHARS_IN_TRACE = 100_000


class Row(BaseModel):
    id: str = Field(min_length=1)
    seed: int
    focal_seat: int = Field(ge=0, lt=10)  # tables seat at least 10 players


class Setup(RunSetup[Row]):
    row_model = Row

    async def load_cases(self, row: Row, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        yield Case(id=row.id, input={"seed": row.seed, "focal_seat": row.focal_seat},
                   expected={"metric": "focal seat's team wins"})


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


def team_of(character: str) -> str:
    from clocktower.characters import DEMONS, MINIONS
    return "evil" if character in (*MINIONS, *DEMONS) else "good"


async def play_game(seed: int, focal_seat: int, game_dir: Path, trace) -> dict:
    # In-process (not a subprocess) so the model calls in clocktower/llm.py join Beaker's trace capture.
    with trace.stage("game", inputs={"seed": seed, "focal_seat": focal_seat, "focal_model": FOCAL_MODEL,
                                     "opponent_models": OPPONENT_MODELS}):
        try:
            await asyncio.to_thread(run_clocktower_cli, [
                "run", "--model", OPPONENT_MODELS, "--focal-seat", str(focal_seat), "--focal-model", FOCAL_MODEL,
                "--seed", str(seed), "--no-probes", "--budget", str(MAX_SPEND_PER_GAME_USD), "--out", str(game_dir),
            ])
        except RuntimeError as game_failure:
            raise RetryableCaseError(f"game failed: {game_failure}") from game_failure
        result = json.loads((game_dir / "result.json").read_text())
        transcript = game_dir / "transcript.md"
        if transcript.exists():
            trace.evidence("transcript", transcript.read_text()[:MAX_TRANSCRIPT_CHARS_IN_TRACE])
        trace.evidence("result", result)
    if result.get("winner") not in ("good", "evil"):
        raise RetryableCaseError(f"game ended without a winner: {result.get('reason')}")
    focal_name = result["focal"]["seat"]
    focal_character = result["final"][focal_name]
    return {"focal_seat": focal_seat, "focal_name": focal_name, "focal_character": focal_character,
            "focal_team": team_of(focal_character), "winner": result["winner"], "reason": result["reason"],
            "days": result["days"], "cost_usd": result["cost"]}


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
    won = float(game["winner"] == game["focal_team"])
    check = Check(
        name="focal seat's team wins", verdict="pass" if won else "fail",
        expected=game["focal_team"], predicted=game["winner"],
        message=f"{game['focal_name']} played {game['focal_character']}: {game['reason']} (day {game['days']})",
    )
    return CaseScore(objective=won, field_scores={"focal_team_win": won}, checks=(check,))


integration = Integration(
    targets=repository(PLAYER_HARNESS_FILES),
    run_setup=Setup, run_case=run_case, score_case=score_case,
)
