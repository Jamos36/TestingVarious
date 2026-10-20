"""Compatibility adapter; new integrations should use service.run and Settings."""
import json
import os
import time

import model_adapter
import transcript_length
from config import Settings
from service import run

STORAGE_MODE = os.getenv('TRANSCRIPT_STORAGE', 'memory')
CHECKPOINT_ROOT = os.getenv('TRANSCRIPT_CHECKPOINT_DIR', './checkpoints')
REVISION = os.getenv('TRANSCRIPT_JOB_REVISION', '1')
MAX_ATTEMPTS = 3


def base_model(*, system_prompt, user_prompt, json_schema):
    payload = json.loads(user_prompt)
    settings = Settings(source_mode=payload['source_mode'], scope=payload['scope'],
        coverage_complete=payload['coverage_complete'], storage=STORAGE_MODE,
        checkpoint_dir=CHECKPOINT_ROOT, job_revision=REVISION,
        retry_attempts=MAX_ATTEMPTS,
        max_request_bytes=transcript_length.MAX_REQUEST_BYTES,
        overlap_turns=transcript_length.OVERLAP_TURNS)
    report = run(payload['transcript'], settings, sources=payload['legal_sources'],
                 model_call=model_adapter.base_model, prompt_text=system_prompt)
    return json.dumps(report['analysis'], ensure_ascii=False)
