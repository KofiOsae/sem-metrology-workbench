from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Measurement:
    name: str
    value: float
    unit: str
    method: str
    confidence: float
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisRecord:
    image_name: str
    image_sha256: str
    mode: str
    calibration: dict[str, Any]
    measurements: list[Measurement]
    settings: dict[str, Any]
    accepted: bool = False
    operator_note: str = ""
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    software_version: str = "0.1.0"

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["measurements"] = [m.as_dict() for m in self.measurements]
        return result
