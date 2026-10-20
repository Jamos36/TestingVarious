"""Compose components here; each component uses ordinary functions or objects."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

from analysis import summarize
from config import Settings
from prompts import PROMPT_VERSION, load_prompt
from schemas import Analysis, LegalSource, SCHEMA_VERSION
from validation import validate_evidence


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        ensure_ascii=False).encode('utf-8')).hexdigest()


def run(transcript, settings=None, *, sources=None, input_name='transcript',
        model_call=None, store_factory=None, progress_sink=None, prompt_text=None):
    """Return a report; nothing is delivered until the caller chooses a writer."""
    settings = settings or Settings()
    sources = sources or []
    if not transcript.strip():
        raise ValueError('Transcript is empty')
    if settings.source_mode == 'model' and sources:
        raise ValueError('Explicit legal sources require supplied mode')
    checked_sources = [LegalSource.model_validate(s).model_dump() for s in sources]
    prompt = prompt_text if prompt_text is not None else load_prompt(settings.prompt_path)
    module_name, function_name = settings.adapter.split(':', 1)
    module = importlib.import_module(module_name)
    model_call = model_call or getattr(module, function_name)
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('test_')}
    model_settings = {name: getattr(module, name, None) for name in
                      ('API_URL', 'MODEL_NAME', 'USE_STRUCTURED_OUTPUT')}
    # Exclude delivery paths/toggles; changing output format need not rescore chunks.
    identity_settings = settings.model_dump(exclude={'output','output_dir','progress',
        'render_markdown','storage','checkpoint_dir','retry_attempts'})
    job_id = digest(dict(transcript=transcript, sources=checked_sources, prompt=prompt,
        schema=Analysis.model_json_schema(), settings=identity_settings,
        model=model_settings, code=code))
    store = None
    if settings.storage != 'none':
        if store_factory is None:
            from job_storage import open_storage
            store_factory = open_storage
        store = store_factory(settings.storage, settings.checkpoint_dir, job_id)
    state = dict(job_id=job_id, status='running', total_chunks=0,
                 completed_chunks=0, checkpoint_bytes=0)

    def progress(**updates):
        state.update(updates)
        if store:
            store.status(state)
        if settings.progress:
            if progress_sink:
                progress_sink(dict(state))
            else:
                print(json.dumps(state), file=sys.stderr, flush=True)

    def validate(raw, payload):
        result = Analysis.model_validate_json(raw)
        validate_evidence(result, payload['transcript'],
            [LegalSource.model_validate(s) for s in payload['legal_sources']],
            payload['coverage_complete'], payload['source_mode'])

    def chunk_call(**request):
        key = digest(request)
        payload = json.loads(request['user_prompt'])
        raw = store.load(key) if store else None
        if raw is None:
            if settings.retry_attempts == 1:
                raw = model_call(**request)
            else:
                from retry_policy import call_with_retry
                raw = call_with_retry(lambda: model_call(**request),
                    attempts=settings.retry_attempts,
                    on_retry=lambda n: progress(status='retrying', attempt=n))
            validate(raw, payload)
            if store:
                store.save(key, raw)
        else:
            validate(raw, payload)
        progress(status='running', completed_chunks=state['completed_chunks']+1,
                 checkpoint_bytes=state['checkpoint_bytes']+len(raw.encode('utf-8')))
        return raw

    def request_call(**request):
        if settings.chunking:
            from transcript_length import process
            return process(**request, model_call=chunk_call,
                max_request_bytes=settings.max_request_bytes,
                overlap_turns=settings.overlap_turns,
                on_plan=lambda n: progress(total_chunks=n))
        # Even when chunking is disabled, oversized input is rejected explicitly.
        if len(json.dumps(request, ensure_ascii=False).encode('utf-8')) > settings.max_request_bytes:
            raise ValueError('Request exceeds configured byte budget; enable chunking or revise the budget')
        progress(total_chunks=1)
        return chunk_call(**request)

    try:
        result = summarize(transcript, request_call, sources=checked_sources,
            coverage_complete=settings.coverage_complete, scope=settings.scope,
            source_mode=settings.source_mode, prompt=prompt)
        progress(status='analysis_completed')
    except Exception:
        progress(status='failed')
        raise
    return dict(schema_version=SCHEMA_VERSION, prompt_version=PROMPT_VERSION,
        job_id=job_id, input_file=input_name, legal_scope=settings.scope,
        source_mode=settings.source_mode, coverage_complete=settings.coverage_complete,
        legal_sources=checked_sources, analysis=result)
