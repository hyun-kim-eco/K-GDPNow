from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Frequency(str, Enum):
    MONTHLY = "M"
    QUARTERLY = "Q"


@dataclass(frozen=True)
class DataSeriesSpec:
    """Metadata to fetch one ECOS series."""

    name: str
    api_code: str
    item_code: str
    frequency: Frequency
    item_code_2: str | None = None
    start_period: str = "199501"
    transform_hint: str | None = None
    release_lag_days: int | None = None
    tags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "api_code": self.api_code,
            "item_code": self.item_code,
            "item_code_2": self.item_code_2 or "",
            "frequency": self.frequency.value,
            "start_period": self.start_period,
            "transform_hint": self.transform_hint or "",
            "release_lag_days": self.release_lag_days,
            "tags": self.tags,
        }
