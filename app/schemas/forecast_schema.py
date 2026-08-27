"""Pydantic v2 schemas for POST /forecast."""

from __future__ import annotations

import math
from datetime import date, timedelta

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _require_finite_non_negative(value: float, field_name: str) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return value


class WeeklyHistoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    week_start_date: date
    week_end_date: date
    week_no: int = Field(..., ge=1, le=53)
    temp_mean: float
    temp_max: float
    temp_min: float
    rainfall_mm: float
    humidity_pct: float
    week_cases: int = Field(..., ge=0)
    cumulative_cases: int = Field(..., ge=0)

    @field_validator(
        "temp_mean",
        "temp_max",
        "temp_min",
        "rainfall_mm",
        "humidity_pct",
    )
    @classmethod
    def weather_finite_non_negative(cls, value: float, info) -> float:
        return _require_finite_non_negative(value, info.field_name)

    @model_validator(mode="after")
    def monday_sunday_week(self) -> "WeeklyHistoryEntry":
        if self.week_start_date.weekday() != 0:
            raise ValueError("week_start_date must be a Monday")
        if self.week_end_date.weekday() != 6:
            raise ValueError("week_end_date must be a Sunday")
        if self.week_end_date != self.week_start_date + timedelta(days=6):
            raise ValueError(
                "week_end_date must be exactly 6 days after week_start_date (Monday–Sunday)"
            )
        return self


class StaticFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    zone_dry_zone: float
    zone_intermediate_zone: float
    zone_wet_zone: float
    population_density: float

    @field_validator("zone_dry_zone", "zone_intermediate_zone", "zone_wet_zone")
    @classmethod
    def zone_must_be_zero_or_one(cls, value: float, info) -> float:
        if not math.isfinite(value) or value not in (0.0, 1.0):
            raise ValueError(f"{info.field_name} must be exactly 0.0 or 1.0")
        return value

    @field_validator("population_density")
    @classmethod
    def density_finite_non_negative(cls, value: float) -> float:
        return _require_finite_non_negative(value, "population_density")

    @model_validator(mode="after")
    def exactly_one_climate_zone(self) -> "StaticFeatures":
        flags = (
            self.zone_dry_zone,
            self.zone_intermediate_zone,
            self.zone_wet_zone,
        )
        if sum(flags) != 1.0:
            raise ValueError(
                "exactly one of zone_dry_zone / zone_intermediate_zone / "
                "zone_wet_zone must be 1.0, the others 0.0"
            )
        return self


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rdhs_id: int = Field(..., ge=0, le=25)
    district_name: str = Field(..., min_length=1)
    target_week_start: date
    static_features: StaticFeatures
    history: list[WeeklyHistoryEntry] = Field(..., min_length=8, max_length=8)

    @model_validator(mode="after")
    def consecutive_history_ending_before_target(self) -> "ForecastRequest":
        if self.target_week_start.weekday() != 0:
            raise ValueError("target_week_start must be a Monday")

        starts = [row.week_start_date for row in self.history]
        if starts != sorted(starts):
            raise ValueError("history must be sorted by week_start_date ascending")

        for i in range(1, len(self.history)):
            expected_start = self.history[i - 1].week_end_date + timedelta(days=1)
            if self.history[i].week_start_date != expected_start:
                raise ValueError(
                    f"history weeks must be consecutive Monday–Sunday with no gaps "
                    f"or overlaps (break before history[{i}])"
                )

        expected_target = self.history[-1].week_end_date + timedelta(days=1)
        if expected_target != self.target_week_start:
            raise ValueError(
                "last history week_end_date must be exactly 1 day before target_week_start"
            )
        return self


class ForecastResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predictions: list[float] = Field(..., min_length=4, max_length=4)
    lower_bounds: list[float] = Field(..., min_length=4, max_length=4)
    upper_bounds: list[float] = Field(..., min_length=4, max_length=4)
    model_version: str
