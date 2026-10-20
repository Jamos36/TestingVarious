"""Independent switches; no hidden dependency between RAG and storage."""
import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    adapter: str = 'model_adapter:base_model'
    source_mode: Literal['supplied', 'model'] = 'model'
    scope: str = 'U.S. Code Title 18'
    coverage_complete: bool = False
    chunking: bool = True
    max_request_bytes: int = Field(default=60_000, ge=1)
    overlap_turns: int = Field(default=2, ge=0)
    storage: Literal['none', 'memory', 'disk'] = 'none'
    checkpoint_dir: str = './checkpoints'
    retry_attempts: int = Field(default=1, ge=1, le=10)
    progress: bool = True
    render_markdown: bool = True
    output: Literal['stdout', 'files'] = 'stdout'
    output_dir: str = './output'
    prompt_path: str | None = None
    job_revision: str = '1'

    @model_validator(mode='after')
    def prevent_nested_wrappers(self):
        if self.adapter.split(':')[0] in {'resumable_adapter', 'transcript_length'}:
            raise ValueError('Configure a raw model adapter; chunking/retries/storage are separate switches')
        if self.source_mode == 'model' and self.coverage_complete:
            raise ValueError('coverage_complete applies only to supplied sources')
        return self


def load_settings(path=None):
    data = json.loads(Path(path).read_text(encoding='utf-8-sig')) if path else {}
    return Settings.model_validate(data)
