"""The focal player's harness: the only code Beaker edits (with playbook.md).

The rules, Storyteller, opponents, scoring and API clients are the eval and live outside this package.
"""
from pathlib import Path

PLAYBOOK = Path(__file__).with_name('playbook.md')


def build_system(base_system: str) -> str:
    """The focal seat's system prompt, given the shared base prompt every seat gets."""
    playbook = PLAYBOOK.read_text().strip()
    return f'{base_system}\n\nPLAYBOOK\n{playbook}' if playbook else base_system


def wrap_turn(text: str, request_kind: str | None) -> str:
    """What the focal seat reads each turn. `request_kind` is night, ravenkeeper, beat, nominate, defense, comment or vote."""
    return text
