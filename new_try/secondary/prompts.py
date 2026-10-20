"""Editable prompt policy; the transcript remains the only interview fact source."""
from pathlib import Path

PROMPT_VERSION = "1.3"
SYSTEM_PROMPT = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")


def load_prompt(path=None):
    return Path(path).read_text(encoding="utf-8") if path else SYSTEM_PROMPT
