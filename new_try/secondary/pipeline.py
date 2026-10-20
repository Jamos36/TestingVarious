"""Command entry point and compatibility imports for earlier integrations."""
from schemas import (Analysis, Concern, Evidence, LegalSource, ModelSource,
                     Participant, Point, Record, Review)
from prompts import SYSTEM_PROMPT
from validation import validate_evidence
def markdown(report):
    from renderers import markdown as render
    return render(report)

from analysis import summarize


def main():
    from cli import main as run_cli
    run_cli()


if __name__ == "__main__":
    main()
