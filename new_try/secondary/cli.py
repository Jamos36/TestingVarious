"""Command-line options override the JSON configuration when explicitly supplied."""
import argparse
import json
from pathlib import Path

from config import load_settings, Settings
from output import deliver
from service import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('transcript', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--adapter')
    parser.add_argument('--source-mode', choices=['model','supplied'])
    parser.add_argument('--sources', type=Path)
    parser.add_argument('--scope')
    parser.add_argument('--coverage-complete', action='store_true', default=None)
    parser.add_argument('--stdout', action='store_true', default=None)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    settings = load_settings(args.config)
    updates = {k: getattr(args, k) for k in
        ('adapter','source_mode','scope','coverage_complete') if getattr(args,k) is not None}
    # Older wrapper names remain valid direct Python adapters, but the new CLI
    # composes their options instead of stacking two copies of those wrappers.
    legacy = updates.get('adapter')
    if legacy in {'transcript_length:base_model', 'resumable_adapter:base_model'}:
        updates['adapter'] = 'model_adapter:base_model'
        updates['chunking'] = True
        if legacy.startswith('resumable_adapter'):
            import os
            updates['storage'] = os.getenv('TRANSCRIPT_STORAGE','memory')
            updates['checkpoint_dir'] = os.getenv('TRANSCRIPT_CHECKPOINT_DIR','./checkpoints')
            updates['job_revision'] = os.getenv('TRANSCRIPT_JOB_REVISION','1')
            updates['retry_attempts'] = 3
    if args.stdout:
        updates['output'] = 'stdout'
    elif args.out:
        updates['output'] = 'files'
    if args.out:
        updates['output_dir'] = str(args.out)
    settings = Settings.model_validate({**settings.model_dump(), **updates})
    if settings.source_mode == 'supplied' and not args.sources:
        parser.error('--sources is required for supplied mode')
    if settings.source_mode == 'model' and args.sources:
        parser.error('--sources applies only to supplied mode')
    sources = json.loads(args.sources.read_text(encoding='utf-8-sig')) if args.sources else []
    report = run(args.transcript.read_text(encoding='utf-8-sig'), settings,
                 sources=sources, input_name=args.transcript.name)
    deliver(report, settings)


if __name__ == '__main__':
    main()
