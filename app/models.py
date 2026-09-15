"""Pydantic schemas for the inspection API."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class InspectionRequest(BaseModel):
    """Request body: the wiring grid plus the exit-id label matrix."""

    model_config = ConfigDict(extra="forbid")

    grid: list[list[str]] = Field(
        description=(
            "Rectangular cell matrix, 1..200 rows by 1..200 columns. "
            "Cells: 'P' power source (exactly one), 'W' wire, 'E' exit light, 'X' empty."
        )
    )
    labels: list[list[str | None]] = Field(
        description=(
            "Matrix with the same shape as `grid`: a unique non-empty exit id "
            "at every 'E' cell, null everywhere else."
        )
    )


class Coordinate(BaseModel):
    """Zero-based cell coordinate."""

    row: int = Field(ge=0)
    col: int = Field(ge=0)


class Failure(BaseModel):
    """One unreachable exit plus the evidence coordinate for its component."""

    exit_id: str
    evidence: Coordinate


class InspectionResponse(BaseModel):
    """PASS when every exit is energized, otherwise FAIL with per-exit evidence."""

    result: Literal["PASS", "FAIL"]
    failures: list[Failure] | None = None


class Issue(BaseModel):
    """A single input-validation problem."""

    code: str
    message: str
    locations: list[Coordinate] = []


class ErrorDetail(BaseModel):
    """Structured rejection payload carried in the `detail` field."""

    code: str
    message: str
    issues: list[Issue] = []


class ErrorResponse(BaseModel):
    detail: ErrorDetail


class DrainState(str, Enum):
    """Admission lifecycle of the service: ACCEPTING -> DRAINING -> DRAINED.

    DRAINED is terminal; only a process restart returns the service to
    ACCEPTING. The same enum is shared by the drain coordinator, the
    admission middleware, and the HTTP layer so every component speaks one
    contract.
    """

    ACCEPTING = "ACCEPTING"
    DRAINING = "DRAINING"
    DRAINED = "DRAINED"


class DrainReport(BaseModel):
    """Outcome of POST /drain.

    The report is produced only after every previously admitted inspection
    has finished, and every concurrent drain caller observes this same
    result.
    """

    state: Literal[DrainState.DRAINED] = Field(
        description="Terminal state; reported only once the service has fully drained."
    )
    in_flight: int = Field(
        ge=0,
        description="Inspections still running when the report was produced; always 0.",
    )


class DrainRejectionDetail(BaseModel):
    """Structured 503 payload for inspections refused after draining began."""

    code: Literal["SERVICE_DRAINING"]
    message: str
    state: DrainState = Field(
        description="Lifecycle state at refusal time: DRAINING or DRAINED."
    )


class DrainRejectionResponse(BaseModel):
    detail: DrainRejectionDetail
