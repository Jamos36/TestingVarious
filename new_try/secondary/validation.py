"""Reject malformed attribution and unsupported source quotations."""
import re

def validate_evidence(analysis, transcript, sources, coverage_complete, source_mode):
    # Supported format: [HH:MM:SS] Speaker: text (brackets optional;
    # MM:SS and fractional seconds are also accepted), plus continuation lines.
    pattern = re.compile(
        r"(?m)^[ \t]*\[?(?P<timestamp>\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)"
        r"\]?[ \t]+(?P<speaker>[^:\n]+):[ \t]*"
    )
    matches = list(pattern.finditer(transcript))
    turns = [(m.group('speaker').strip(), m.group('timestamp'),
              transcript[m.start():matches[i+1].start() if i+1 < len(matches) else len(transcript)])
             for i, m in enumerate(matches)]

    def check(e):
        if not turns:
            raise ValueError(
                "Cannot validate attribution: normalize transcript to "
                "[HH:MM:SS] Speaker: text with continuation lines")
        if not any(speaker == e.speaker and
                   (e.timestamp is None or timestamp == e.timestamp) and
                   e.quote in text for speaker, timestamp, text in turns):
            raise ValueError("Evidence quote does not match its speaker/timestamp turn")
    if turns:
        labels = {speaker for speaker, _, _ in turns}
        if any(p.speaker not in labels for p in analysis.participants):
            raise ValueError("Participant label is not in transcript turns")
    for point in analysis.high_points + analysis.timeline:
        for e in point.evidence:
            check(e)
    if source_mode == "supplied" and analysis.model_sources:
        raise ValueError("Supplied mode cannot use model sources")
    selected = sources if source_mode == "supplied" else analysis.model_sources
    by_id = {s.source_id: s for s in selected}
    if len(by_id) != len(selected):
        raise ValueError("Duplicate source IDs")
    for source in analysis.model_sources:
        if source.origin == "model_memory" and any(
                x is not None for x in (source.text, source.source_url, source.version)):
            raise ValueError("Memory references cannot claim retrieved text/metadata")
        if source.origin == "model_rag" and not source.text:
            raise ValueError("RAG references require retrieved text")
    for concern in analysis.legal_review.concerns:
        source = by_id.get(concern.source_id)
        if source is None:
            raise ValueError("Unknown legal source ID")
        if source_mode == "model" and source.origin == "model_memory":
            if concern.legal_quote is not None:
                raise ValueError("Memory references must not claim verified legal quotations")
        elif not concern.legal_quote or concern.legal_quote not in source.text:
            raise ValueError("Legal quotation is not grounded in source text")
        for e in concern.evidence + concern.counterevidence:
            check(e)
    if analysis.legal_review.status == "none_identified":
        if not selected or (source_mode == "supplied" and not coverage_complete):
            raise ValueError("Negative screening requires references and supplied-mode coverage")
    if source_mode == "model":
        analysis.legal_review.limitations.append(
            "Model-provided legal references and retrieval coverage are not independently verified by this pipeline.")
        if any(s.origin == "model_memory" for s in analysis.model_sources):
            analysis.legal_review.limitations.append(
                "Screening uses unverified model memory; legal accuracy, currency and completeness are unconfirmed.")

