"""One-request analysis with interchangeable model access."""
import json
from schemas import Analysis, LegalSource
from prompts import SYSTEM_PROMPT
from validation import validate_evidence

def summarize(transcript, base_model, *, sources=None, coverage_complete=False,
              scope="U.S. Code Title 18", source_mode="supplied", prompt=None):
    """base_model(system_prompt=..., user_prompt=..., json_schema=...) -> str.

    supplied mode uses explicit sources; model mode uses the adapter's RAG/knowledge.
    The adapter must reject truncation/refusals and must not silently trim input.
    """
    if not transcript.strip():
        raise ValueError("Transcript is empty")
    if source_mode not in {"supplied", "model"}:
        raise ValueError("source_mode must be supplied or model")
    if source_mode == "model" and (sources or coverage_complete):
        raise ValueError("Sources/coverage_complete apply only to supplied mode")
    sources = [LegalSource.model_validate(s) for s in (sources or [])]
    if len({s.source_id for s in sources}) != len(sources):
        raise ValueError("Duplicate legal source IDs")
    payload = {"source_mode": source_mode, "scope": scope, "coverage_complete": coverage_complete,
               "legal_sources": [s.model_dump() for s in sources], "transcript": transcript}
    raw = base_model(system_prompt=prompt or SYSTEM_PROMPT,
                     user_prompt=json.dumps(payload, ensure_ascii=False),
                     json_schema=Analysis.model_json_schema())
    analysis = Analysis.model_validate_json(raw)
    validate_evidence(analysis, transcript, sources, coverage_complete, source_mode)
    return analysis.model_dump()

