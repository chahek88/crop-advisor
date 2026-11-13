from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class PlotIn(BaseModel):
    id: str | None = Field(None, description="Optional slug; derived from the name if omitted", max_length=64)
    name: str = Field(..., min_length=1, max_length=120)
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    area_ha: float = Field(1.0, gt=0, le=100000)


class SensorReading(BaseModel):
    """One message from a field sensor / gateway. Send whatever your hardware measures."""

    plot_id: str
    device_id: str | None = Field(None, max_length=64)
    timestamp: datetime | None = Field(None, description="ISO-8601; server time is used when omitted")
    n: float | None = Field(None, ge=0, le=5000, description="Nitrogen")
    p: float | None = Field(None, ge=0, le=5000, description="Phosphorus")
    k: float | None = Field(None, ge=0, le=5000, description="Potassium")
    ph: float | None = Field(None, ge=0, le=14)
    soil_moisture: float | None = Field(None, ge=0, le=100, description="Volumetric water content, %")
    soil_temp: float | None = Field(None, ge=-30, le=80, description="degC")
    ec: float | None = Field(None, ge=0, le=100, description="Electrical conductivity, dS/m")

    @model_validator(mode="after")
    def _has_value(self):
        if all(getattr(self, f) is None for f in ("n", "p", "k", "ph", "soil_moisture", "soil_temp", "ec")):
            raise ValueError("reading contains no measurements")
        return self


class ManualInputs(BaseModel):
    """Optional what-if overrides (take priority over sensors / weather)."""

    N: float | None = Field(None, ge=0, le=500)
    P: float | None = Field(None, ge=0, le=500)
    K: float | None = Field(None, ge=0, le=500)
    temperature: float | None = Field(None, ge=-10, le=60)
    humidity: float | None = Field(None, ge=0, le=100)
    ph: float | None = Field(None, ge=0, le=14)
    rainfall: float | None = Field(None, ge=0, le=3000)


class AdhocRequest(BaseModel):
    """Prediction without a registered plot: manual values and/or a location for live weather."""

    inputs: ManualInputs = ManualInputs()
    lat: float | None = Field(None, ge=-90, le=90)
    lon: float | None = Field(None, ge=-180, le=180)
    crop: str | None = None
    top_k: int = Field(3, ge=1, le=10)
