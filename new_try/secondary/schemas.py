"""Stable model output contract. Version changes belong here."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.1"

class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Evidence(Record):
    speaker: str
    timestamp: str | None
    quote: str = Field(min_length=1)


class Point(Record):
    summary: str
    evidence: list[Evidence] = Field(min_length=1)


class Participant(Record):
    speaker: str
    role: str | None


class LegalSource(Record):
    source_id: str
    citation: str
    source_url: str | None
    version: str | None
    text: str = Field(min_length=1)


class ModelSource(Record):
    source_id: str
    citation: str
    origin: Literal["model_rag", "model_memory"]
    source_url: str | None
    version: str | None
    text: str | None


class Concern(Record):
    description: str
    source_id: str
    legal_quote: str | None
    evidence: list[Evidence] = Field(min_length=1)
    element_analysis: str
    missing_facts: list[str]
    counterevidence: list[Evidence]


class Review(Record):
    status: Literal["potential_concern", "none_identified", "insufficient_information"]
    explanation: str
    concerns: list[Concern]
    limitations: list[str]

    @model_validator(mode="after")
    def consistent_status(self):
        if bool(self.concerns) != (self.status == "potential_concern"):
            raise ValueError("Concerns must be present exactly when status is potential_concern")
        return self


class Analysis(Record):
    participants: list[Participant]
    summary: str = Field(min_length=1)
    high_points: list[Point]
    timeline: list[Point]
    unresolved_questions: list[str]
    legal_review: Review
    model_sources: list[ModelSource]


