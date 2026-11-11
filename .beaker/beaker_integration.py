"""Beaker integration: hillclimb clocktower/playbook.md, advice given to one team of gpt-6-luna players.

A case is one deal (seed). It plays two games on that deal, every seat gpt-6-luna: one where the
good team gets the playbook, one where the evil team does. The other team gets no playbook.
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

PLAYER_MODEL = "gpt-6-luna"
CHECKOUT_ROOT = Path(__file__).resolve().parent.parent
PLAYBOOK_TEAMS = ("good", "evil")
MAX_SPEND_PER_GAME_USD = 25


class Row(BaseModel):
    id: str = Field(min_length=1)
    seed: int


class Setup(RunSetup[Row]):
    row_model = Row

    async def load_cases(self, row: Row, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        yield Case(id=row.id, input={"seed": row.seed}, expected={"metric": "playbook team wins"})


async def play_game_with_playbook_for(team: str, seed: int, game_dir: Path) -> dict:
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "clocktower", "run",
        "--model", PLAYER_MODEL, "--seed", str(seed), "--playbook-team", team,
        "--no-probes", "--budget", str(MAX_SPEND_PER_GAME_USD), "--out", str(game_dir),
        cwd=CHECKOUT_ROOT, env={**os.environ, "PYTHONPATH": str(CHECKOUT_ROOT)},
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    try:
        output, _ = await process.communicate()
    except asyncio.CancelledError:
        process.kill()
        await process.wait()
        raise
    result_path = game_dir / "result.json"
    if process.returncode != 0 or not result_path.exists():
        raise RetryableCaseError(f"{team}-playbook game failed (exit {process.returncode}): {output.decode()[-800:]}")
    result = json.loads(result_path.read_text())
    if result.get("winner") not in ("good", "evil"):
        raise RetryableCaseError(f"{team}-playbook game ended without a winner: {result.get('reason')}")
    return {"winner": result["winner"], "reason": result["reason"], "days": result["days"], "cost_usd": result["cost"]}


async def run_case(*, case_input: dict, runtime: RolloutRuntime) -> CaseResult:
    if not (CHECKOUT_ROOT / "clocktower" / "playbook.md").exists():
        raise RuntimeError(f"clocktower/playbook.md missing under {CHECKOUT_ROOT}")
    seed = case_input["seed"]
    with tempfile.TemporaryDirectory(prefix="clocktower-games-") as temp_dir:
        games = await asyncio.gather(*(
            play_game_with_playbook_for(team, seed, Path(temp_dir) / team) for team in PLAYBOOK_TEAMS
        ))
    return CaseResult(output=dict(zip(PLAYBOOK_TEAMS, games)), output_kind="record")


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    field_scores, checks = {}, []
    for team in PLAYBOOK_TEAMS:
        game = result.output[team]
        won = float(game["winner"] == team)
        field_scores[f"{team}_playbook_win"] = won
        checks.append(Check(
            name=f"{team} team with the playbook wins", verdict="pass" if won else "fail",
            expected=team, predicted=game["winner"], message=f"{game['reason']} (day {game['days']})",
        ))
    return CaseScore(objective=sum(field_scores.values()) / len(field_scores),
                     field_scores=field_scores, checks=tuple(checks))


integration = Integration(
    targets=repository(("clocktower/playbook.md",)),
    run_setup=Setup, run_case=run_case, score_case=score_case,
)
