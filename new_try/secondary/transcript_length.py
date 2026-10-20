"""Length-aware adapter. Use --adapter transcript_length:base_model."""
import json
import re

def call_model(**kwargs):
    from model_adapter import base_model
    return base_model(**kwargs)

from schemas import Analysis, LegalSource
from validation import validate_evidence

# Tune for your company model. This is a BYTE budget, not an exact token count.
# It includes prompts, supplied sources, schema and JSON encoding. Leave room in
# the real context window for output, server instructions and internal RAG text.
MAX_REQUEST_BYTES = 60_000
OVERLAP_TURNS = 2

TURN_START = re.compile(r"(?m)^(?=\s*\[?\d{1,2}:\d{2}(?::\d{2})?)")
CHUNK_INSTRUCTIONS = """
This request is ONE SECTION of a longer interview, with possible overlap.
Summarize all material in this section. Preserve names, dates, amounts, denials,
qualifications, relevant quotations and unresolved references to other sections.
Do not present this section as the entire interview or claim complete coverage.
Keep evidence excerpts concise and verbatim. Screen the section using the
requested source mode. Other sections may supply missing context.
"""


def encode(value):
    return json.dumps(value, ensure_ascii=False)


def unique(items):
    """Stable deduplication; no model call or paraphrasing."""
    result, seen = [], set()
    for item in items:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def split_transcript(text, fits, overlap_turns=None):
    """Keep timestamped turns intact; fall back to lines without timestamps."""
    overlap = OVERLAP_TURNS if overlap_turns is None else overlap_turns
    starts = [match.start() for match in TURN_START.finditer(text)]
    if starts:
        boundaries = sorted(set([0, *starts, len(text)]))
        turns = [text[a:b] for a, b in zip(boundaries, boundaries[1:])]
    else:
        turns = text.splitlines(keepends=True)
    chunks, current = [], []
    for turn in turns:
        if not fits(turn):
            raise ValueError(
                "One transcript turn exceeds the request budget. Increase the "
                "budget within your model limit, or split that turn into labeled "
                "continuations preserving its speaker/timestamp. Nothing was trimmed."
            )
        if current and not fits(''.join(current) + turn):
            chunks.append(''.join(current))
            current = current[-overlap:] if overlap else []
            while current and not fits(''.join(current) + turn):
                current.pop(0)
        current.append(turn)
    if current:
        chunks.append(''.join(current))
    return chunks


def base_model(*, system_prompt, user_prompt, json_schema):
    return process(system_prompt=system_prompt, user_prompt=user_prompt,
                   json_schema=json_schema, model_call=call_model)


def process(*, system_prompt, user_prompt, json_schema, model_call, on_plan=None,
            max_request_bytes=None, overlap_turns=None):
    budget = MAX_REQUEST_BYTES if max_request_bytes is None else max_request_bytes
    overlap = OVERLAP_TURNS if overlap_turns is None else overlap_turns
    payload = json.loads(user_prompt)
    transcript = payload['transcript']

    def fits(text, system=system_prompt):
        request = {
            'system_prompt': system,
            'user_prompt': encode({**payload, 'transcript': text}),
            'json_schema': json_schema,
        }
        return len(encode(request).encode('utf-8')) <= budget

    if fits(transcript):
        if on_plan:
            on_plan(1)
        return model_call(system_prompt=system_prompt, user_prompt=user_prompt,
                          json_schema=json_schema)

    chunk_system = system_prompt + CHUNK_INSTRUCTIONS
    if not fits('', chunk_system):
        raise ValueError(
            "Instructions, schema and legal sources alone exceed the request "
            "budget. Retrieve fewer relevant legal excerpts or use a larger context."
        )
    chunks = split_transcript(transcript, lambda t: fits(t, chunk_system), overlap)
    if on_plan:
        on_plan(len(chunks))
    analyses = []
    sources = [LegalSource.model_validate(s) for s in payload['legal_sources']]
    for number, chunk in enumerate(chunks, 1):
        raw = model_call(
            system_prompt=chunk_system,
            user_prompt=encode({**payload, 'transcript': chunk}),
            json_schema=json_schema,
        )
        analysis = Analysis.model_validate_json(raw)
        validate_evidence(analysis, chunk, sources,
                          payload['coverage_complete'], payload['source_mode'])
        data = analysis.model_dump()
        # A RAG agent may reuse IDs such as "source1" on separate requests.
        # Namespace each chunk's returned IDs to prevent citation collisions.
        ids = {s['source_id']: f"chunk{number}:{s['source_id']}"
               for s in data['model_sources']}
        for source in data['model_sources']:
            source['source_id'] = ids[source['source_id']]
        for concern in data['legal_review']['concerns']:
            concern['source_id'] = ids.get(concern['source_id'], concern['source_id'])
        analyses.append(data)

    # Merge in Python: there is no oversized final model request and no concern
    # or quotation is dropped to fit an artificial final response token limit.
    result = analyses[0].copy()
    result['summary'] = '\n\n'.join(
        f"Section {i}: {a['summary']}" for i, a in enumerate(analyses, 1)
    )
    roles = {}
    for a in analyses:
        for p in a['participants']:
            roles.setdefault(p['speaker'], set())
            if p['role'] is not None:
                roles[p['speaker']].add(p['role'])
    result['participants'] = [
        {'speaker': speaker, 'role': next(iter(r)) if len(r) == 1 else None}
        for speaker, r in roles.items()
    ]
    for key in ('high_points', 'timeline', 'unresolved_questions', 'model_sources'):
        result[key] = unique([item for a in analyses for item in a[key]])
    concerns = [c for a in analyses for c in a['legal_review']['concerns']]
    limitations = [x for a in analyses for x in a['legal_review']['limitations']]
    limitations.append(
        f"Processed {len(chunks)} overlapping transcript sections. Summaries were "
        "combined in Python; cross-section legal relationships and contradictions "
        "were not assessed globally. Overlap may produce repeated points."
    )
    result['legal_review'] = {
        'status': 'potential_concern' if concerns else 'insufficient_information',
        'explanation': (
            'Potential concerns were identified during section-by-section screening.'
            if concerns else
            'No potential concern was identified within individual sections. '
            'A whole-interview legal assessment is still required.'
        ),
        'concerns': concerns,
        'limitations': unique(limitations),
    }
    # Validate merged citations and quotations against the complete original.
    checked = Analysis.model_validate(result)
    validate_evidence(checked, transcript, sources,
                      payload['coverage_complete'], payload['source_mode'])
    return checked.model_dump_json()
