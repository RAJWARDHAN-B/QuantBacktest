from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class SourceRecord(BaseModel):
    """One retrieved external resource, for the run manifest."""

    url: str
    retrieved_at: datetime
    from_cache: bool
