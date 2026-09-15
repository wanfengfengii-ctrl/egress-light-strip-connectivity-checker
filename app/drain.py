"""Drain coordination for rolling updates.

While ACCEPTING, the service admits inspections. POST /drain flips the
coordinator to DRAINING — new inspections are refused with a structured
503 — and waits until every previously admitted inspection has finished
before reporting DRAINED. DRAINED is terminal: only a process restart
restores ACCEPTING.

The coordinator is a single-process state machine. One lock guards the
state and the in-flight counter so admission and draining form an atomic
boundary: a request is either counted before the transition (and the drain
waits for it) or refused after it. ``_empty`` is set exactly while the
in-flight count is zero, so drain waiters block without polling and can
never miss a wakeup.
"""
from __future__ import annotations

import asyncio
import threading

from starlette import status
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.models import DrainReport, DrainRejectionDetail, DrainState

_INSPECT_PATH = "/inspect"

_REJECTION_MESSAGES = {
    DrainState.DRAINING: (
        "The service is draining for a rolling update and no longer "
        "accepts new inspection requests."
    ),
    DrainState.DRAINED: (
        "The service has drained and no longer accepts inspection "
        "requests; only a process restart restores admission."
    ),
}


class DrainCoordinator:
    """Tracks the admission state and the in-flight inspection count."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = DrainState.ACCEPTING
        self._in_flight = 0
        self._empty = threading.Event()
        self._empty.set()

    @property
    def state(self) -> DrainState:
        with self._lock:
            return self._state

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    def try_admit(self) -> bool:
        """Admit one inspection, or refuse once draining has begun.

        The state check and the counter increment are one atomic step, so
        an admission never slips in after the drain transition started.
        """
        with self._lock:
            if self._state is not DrainState.ACCEPTING:
                return False
            self._in_flight += 1
            self._empty.clear()
            return True

    def release(self) -> None:
        """Mark one admitted inspection as finished.

        Must be called exactly once per successful ``try_admit`` — from a
        ``finally`` path — so success, rejection, error, and cancellation
        all free the drain waiter.
        """
        with self._lock:
            if self._in_flight <= 0:
                raise RuntimeError("release() without a matching try_admit()")
            self._in_flight -= 1
            if self._in_flight == 0:
                self._empty.set()

    def rejection_detail(self) -> DrainRejectionDetail:
        """Structured 503 payload describing why admission was refused."""
        state = self.state
        message = _REJECTION_MESSAGES.get(state, _REJECTION_MESSAGES[DrainState.DRAINED])
        return DrainRejectionDetail(
            code="SERVICE_DRAINING",
            message=message,
            state=state,
        )

    async def drain(self) -> DrainReport:
        """Transition to DRAINING and wait for in-flight inspections.

        Concurrent callers share the one transition: the state flips at
        most once, everyone waits on the same emptiness signal, and all
        observe the identical terminal report. Awaiting the threading
        event in an executor keeps the event loop free while the last
        inspections finish.
        """
        with self._lock:
            if self._state is DrainState.ACCEPTING:
                self._state = DrainState.DRAINING
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._empty.wait)
        with self._lock:
            # No new admission can arrive once the state left ACCEPTING, so
            # the count can only have reached zero and stayed there.
            self._state = DrainState.DRAINED
            return DrainReport(state=DrainState.DRAINED, in_flight=self._in_flight)


class InspectAdmissionMiddleware:
    """Pure-ASGI gate: admits POST /inspect only while ACCEPTING.

    Every other route (health checks, docs, /drain itself) passes through
    untouched. The in-flight count is released in a ``finally`` block, so
    every outcome — PASS, FAIL, 422 rejection, internal error, or client
    cancellation — frees the drain waiter exactly once.
    """

    def __init__(self, app: ASGIApp, coordinator: DrainCoordinator) -> None:
        self.app = app
        self.coordinator = coordinator

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] != _INSPECT_PATH
        ):
            await self.app(scope, receive, send)
            return
        if not self.coordinator.try_admit():
            detail = self.coordinator.rejection_detail()
            response = JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": detail.model_dump(mode="json")},
            )
            await response(scope, receive, send)
            return
        try:
            await self.app(scope, receive, send)
        finally:
            self.coordinator.release()
