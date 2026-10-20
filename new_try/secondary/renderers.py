"""Render validated data without asking the model to rewrite it."""
def markdown(report):
    """Render the validated data, never ask the LLM for a second representation."""
    def safe(value):
        value = str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        for char in "\\`*_{}[]()#+!|":
            value = value.replace(char, "\\" + char)
        return value.replace("\n", " ")
    a = report["analysis"]
    lines = ["# Interview summary", "", f"Source: {safe(report['input_file'])}",
             "", "## Participants", ""]
    lines += [f"- {safe(p['speaker'])}: {safe(p['role'] or 'Role not stated')}" for p in a['participants']]
    lines += ["", "## Summary", "", safe(a['summary'])]
    def evidence(items):
        for e in items:
            lines.append(f"  - {safe(e['timestamp'] or 'Timestamp unavailable')} | {safe(e['speaker'])}: {safe(e['quote'])}")
    for title, key in [("High points", "high_points"), ("Timeline", "timeline")]:
        lines.extend(["", f"## {title}", ""])
        for p in a[key]:
            lines.append(f"- {safe(p['summary'])}")
            evidence(p['evidence'])
        if not a[key]:
            lines.append("None identified.")
    review = a['legal_review']
    lines += ["", "## Legal screening", "", f"Status: {safe(review['status'])}", "", safe(review['explanation'])]
    for i, c in enumerate(review['concerns'], 1):
        lines += ["", f"### Potential concern {i}", "", safe(c['description']), "",
                  f"Source: {safe(c['source_id'])}", "", f"Legal excerpt: {safe(c['legal_quote'] or 'Unavailable; memory-based reference')}",
                  "", safe(c['element_analysis']), "", "Evidence:"]
        evidence(c['evidence'])
        lines += ["", "Counterevidence:"]
        evidence(c['counterevidence'])
        lines += ["", "Missing facts:"] + [f"- {safe(x)}" for x in c['missing_facts']]
    for title, items in [("Limitations", review['limitations']), ("Unresolved questions", a['unresolved_questions'])]:
        lines += ["", f"## {title}", ""] + ([f"- {safe(x)}" for x in items] or ["None stated."])
    lines += ["", "## Reviewed legal sources", ""]
    for s in report['legal_sources'] + a['model_sources']:
        lines += [f"- {safe(s['source_id'])}: {safe(s['citation'])}; version: {safe(s['version'] or 'unspecified')}; URL: {safe(s['source_url'] or 'unavailable')}"]
        lines.append(f"  - Origin: {safe(s.get('origin', 'supplied_document'))}")
    return "\n".join(lines) + "\n"

