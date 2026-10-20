# Modular interview summarization pipeline

The transcript supplies interview facts. Legal references supply law, never missing interview facts. The LLM writes an Analysis object; Python validates evidence, assembles chunks and renders outputs.

## Start

Python 3.10+; install `python -m pip install -r requirements.txt`.
Edit model_adapter.py for your company's URL, model, authentication and RAG settings. The generic adapter uses an OpenAI-compatible HTTP shape; it does not automatically enable company RAG.

```bash
python pipeline.py interview.txt --config configs/memory.json
```

This preset uses company RAG/model knowledge, chunking and up to three transient attempts, with no checkpoint/result files. It returns one JSON envelope on stdout containing report and markdown. Progress goes to stderr. The preset name means processing in RAM; storage="none" avoids duplicate memory checkpoints.

```bash
python pipeline.py interview.txt --config configs/resume.json
python pipeline.py interview.txt --config configs/minimal.json
```

resume uses disk checkpoints; edit checkpoint_dir to an approved persistent location. Rerunning the same input/config/model revision resumes saved chunks. minimal disables chunking, storage, retries, Markdown and progress. Oversized requests fail explicitly.

## Supplied legal text

```bash
python pipeline.py interview.txt --config configs/memory.json --source-mode supplied --sources legal_sources.json
```

The source JSON is an array of objects with source_id, citation, source_url (string/null), version (string/null), and text. Whole text or relevant excerpts can be supplied; PDF/Word extraction is not included. The bundled source JSON is empty. Add --coverage-complete only after verifying coverage of the configured legal review scope.

## Independent switches

config.py defines strict Settings. Unknown keys, string booleans and invalid values fail. Config paths and output paths are relative to the working directory. CLI flags override supplied configuration fields. Default source mode is model and default output is stdout. Use --out output/interview-id for files; --stdout takes precedence when both are present.

| Setting | Options |
| --- | --- |
| adapter | module:function with the three-argument model contract |
| source_mode | model or supplied |
| scope | Defaults to U.S. Code Title 18 |
| chunking | true or false |
| max_request_bytes | Estimated request size budget, not exact tokens |
| overlap_turns | Previous turns repeated between chunks |
| storage | none, memory or disk |
| retry_attempts | 1 disables retries; up to 10 attempts allowed |
| progress | JSON progress on stderr or an injected callback |
| render_markdown | true or false |
| output | stdout or files |
| output_dir | Final result directory in files mode |
| checkpoint_dir | Persistent checkpoint location in disk mode |
| prompt_path | Custom instruction file or null for prompt.txt |
| job_revision | Change when backend model/RAG knowledge changes |

Settings also includes coverage_complete, which applies only to supplied mode. Do not put API credentials in the settings file. LLM_API_KEY, LLM_API_URL and LLM_MODEL configure the bundled raw adapter through the environment. USE_STRUCTURED_OUTPUT is edited in model_adapter.py; set it false for prompt-based schema fallback if needed.

## Modules

service.py composes the components and returns a report. analysis.py makes one validated request. schemas.py owns the Pydantic contract. prompt.txt is the actual default instruction policy, loaded by prompts.py. validation.py checks quotes against speaker/timestamp turns and checks legal references. transcript_length.py is optional chunking/assembly. job_storage.py is optional memory/disk checkpoint storage. retry_policy.py is optional transport retry logic. renderers.py creates Markdown. output.py controls delivery. cli.py owns arguments; pipeline.py preserves earlier Python imports and command entry.

For new Python integrations:

```python
from config import Settings
from service import run

settings = Settings(chunking=True, storage="none", retry_attempts=3)
report = run(transcript_text, settings, input_name="interview.txt")
# Send report to your existing backend here.
```

Inject a model callable with model_call=..., storage factory with store_factory=..., or progress callback with progress_sink=.... Storage objects provide load(key), save(key, raw_json), status(progress_dict). A custom sink/database/queue can implement those methods. Model callables accept system_prompt, user_prompt and json_schema keywords and return final JSON text. They must reject truncation/refusal and handle company tool execution before returning. Modules use short comments at important boundaries.

## Behavior and limitations

Long transcripts are summarized section by section and assembled in Python; there is no whole-interview LLM synthesis. All section concerns are retained, including ones that later context may qualify. If no section identifies a concern, chunked overall status is insufficient_information, not none_identified. Exact duplicate points are removed; paraphrased duplicates may remain.

Evidence format is [HH:MM:SS] Speaker: text with optional brackets, MM:SS or fractional seconds and continuation lines. Evidence checks fail on unsupported attribution formats. The code cannot prove every narrative assertion is entailed by the transcript; review paraphrases and omissions. Model knowledge and legal documents are not allowed to create interview facts.

The byte budget is a heuristic and excludes hidden company RAG/tool context. Oversized individual turns or fixed legal sources raise errors. Provider token counting is needed for exact limits. Input and accumulated results remain in RAM; this is not a streaming, bounded-memory implementation.

Disk checkpointing saves validated responses, not raw transcript copies; responses can contain quotations and retrieved legal text. A job hash separates changed input/prompt/schema/model/settings. Set job_revision when server-side changes are hidden. Code changes also invalidate checkpoints. Reusing checkpoints is not an exactly-once remote request guarantee. Run one worker per identical job. No distributed locks, queue, database, remote dashboard delivery, cleanup policy or encryption is bundled. Status analysis_completed is not a dashboard delivery acknowledgment.

stdout delivery avoids result files. File output writes summary.json and optional summary.md atomically per file, not as an atomic pair. Prefer a unique output directory per job and only consume outputs after successful completion. Disabling Markdown does not delete a preexisting Markdown file.

The local test suite uses mock models only. Run `python -m unittest discover -v` from this folder. No real proprietary transcript, company endpoint or RAG library was tested. See the Word guide for design decisions, replacement contracts, configuration, recovery scenarios and risks.
