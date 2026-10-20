# Optional processing and storage

Use the JSON presets for new integrations. They choose analysis features independently of legal source mode.

```bash
python pipeline.py interview.txt --config configs/memory.json
python pipeline.py interview.txt --config configs/resume.json
python pipeline.py interview.txt --config configs/minimal.json
```

- memory.json uses RAM processing with storage="none", chunking, three total transient attempts, progress on stderr and JSON plus Markdown on stdout. It creates no transcript checkpoint or final result files.
- resume.json uses storage="disk" and an editable checkpoint_dir. Completed validated responses can be reused after interruption.
- minimal.json disables chunking, checkpointing, retries, progress and Markdown; it returns JSON on stdout and rejects oversized input.

All presets default to source_mode="model": use available company RAG and model memory. The raw model adapter must configure the actual company endpoint/tool access. No supplied legal file is required. Select --source-mode supplied --sources legal_sources.json to use explicit legal text instead. Storage and legal source selection do not depend on one another.

## Optional modules

Set chunking=false to bypass transcript_length.py, retry_attempts=1 to bypass retry_policy.py, storage="none" to bypass checkpoint storage, or render_markdown=false to bypass renderers.py. job_storage.py is still needed if file output is selected. The new service lazily imports optional modules. Do not delete core modules or the configured raw adapter. Use service.run directly to bypass CLI delivery entirely.

## Resume and storage limits

Disk checkpoints contain validated responses, not duplicate full transcript inputs; they can contain sensitive quotes and legal excerpts. Sizes are variable. checkpoint_bytes reports cumulative response payload bytes for completed calls in the current execution, including reused responses. It is not a filesystem quota or total storage meter. No automatic cleanup, encryption, retention or distributed locking is included. Use one worker per identical job directory.

The job ID accounts for transcript/source/prompt/schema content, processing choices, visible model settings and code fingerprints. Set job_revision when hidden model or RAG content changes. Re-run the same job and persistent path to revalidate and reuse saved chunks. Lost unsaved responses can require repeat remote work. Disk-full, invalid checkpoints and denied writes fail visibly; there is no silent storage fallback.

Memory mode does not survive process death. storage="memory" provides an in-process dictionary cache but can duplicate RAM use. storage="none" is the default and avoids that extra cache. Both still hold the transcript and accumulated analysis in RAM; neither is a bounded-memory streaming implementation.

## Delivery and progress

output="stdout" returns one JSON envelope with report and optional markdown. The caller must capture/forward it. output="files" or --out DIRECTORY writes summary.json and optional summary.md. Individual files are atomic; the pair is not a transaction. Use unique job directories and consume results only after successful completion. Disabling Markdown does not remove an existing Markdown file.

Progress is JSON on stderr or an injected progress_sink; disk mode also saves status.json. Stages include running, retrying, failed and analysis_completed. The last means analysis finished, not that a dashboard received it. No dashboard network call, background worker, queue or database is provided.

## Earlier adapter commands

Older --adapter transcript_length:base_model and --adapter resumable_adapter:base_model commands are translated by cli.py into the new settings. The resumable compatibility command reads TRANSCRIPT_STORAGE, TRANSCRIPT_CHECKPOINT_DIR and TRANSCRIPT_JOB_REVISION from the environment. Do not put wrapper adapters into a new JSON config; adapter must identify the raw company model function.

See the Word guide for the complete module map, design rationale, extension contracts, configuration reference and risks.
