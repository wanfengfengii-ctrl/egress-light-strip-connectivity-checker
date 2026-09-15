"""Drain lifecycle: admission stops, in-flight inspections finish, and the
terminal DRAINED state holds until the process restarts."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

import app.main as main_module
from app.drain import DrainCoordinator, InspectAdmissionMiddleware
from app.inspector import inspect as real_inspect
from app.main import create_app
from app.models import DrainState

PASS_PAYLOAD = {
    "grid": [
        ["P", "W", "E"],
        ["X", "X", "W"],
        ["E", "W", "W"],
    ],
    "labels": [
        [None, None, "EXIT-01"],
        [None, None, None],
        ["EXIT-02", None, None],
    ],
}

# Two power sources: always rejected with a structured 422.
INVALID_PAYLOAD = {"grid": [["P", "W", "P"]], "labels": [[None, None, None]]}

DRAINED_BODY = {"state": "DRAINED", "in_flight": 0}


def _eventually(predicate, timeout=5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _build():
    coordinator = DrainCoordinator()
    return coordinator, create_app(coordinator)


# ---------------------------------------------------------------------------
# Coordinator: the single-process state machine
# ---------------------------------------------------------------------------


def test_coordinator_admits_and_releases_while_accepting():
    coordinator = DrainCoordinator()
    assert coordinator.state is DrainState.ACCEPTING
    assert coordinator.try_admit()
    assert coordinator.in_flight == 1
    coordinator.release()
    assert coordinator.in_flight == 0
    assert coordinator.state is DrainState.ACCEPTING


def test_release_without_admission_is_rejected():
    coordinator = DrainCoordinator()
    with pytest.raises(RuntimeError):
        coordinator.release()


def test_drain_on_idle_service_is_immediate_and_terminal():
    coordinator = DrainCoordinator()
    report = asyncio.run(coordinator.drain())
    assert report.state is DrainState.DRAINED
    assert report.in_flight == 0
    assert coordinator.state is DrainState.DRAINED
    assert not coordinator.try_admit()


def test_drain_waits_for_inflight_release():
    coordinator = DrainCoordinator()
    coordinator.try_admit()

    async def scenario():
        task = asyncio.create_task(coordinator.drain())
        await asyncio.sleep(0.05)
        assert not task.done(), "drain finished while an inspection was in flight"
        assert coordinator.state is DrainState.DRAINING
        coordinator.release()
        return await asyncio.wait_for(task, timeout=1)

    report = asyncio.run(scenario())
    assert (report.state, report.in_flight) == (DrainState.DRAINED, 0)


def test_concurrent_drains_share_one_transition_and_result():
    coordinator = DrainCoordinator()
    coordinator.try_admit()

    async def scenario():
        tasks = [asyncio.create_task(coordinator.drain()) for _ in range(5)]
        await asyncio.sleep(0.05)
        assert all(not task.done() for task in tasks)
        coordinator.release()
        return await asyncio.wait_for(asyncio.gather(*tasks), timeout=1)

    reports = asyncio.run(scenario())
    assert len(reports) == 5
    assert all(report == reports[0] for report in reports)
    assert (reports[0].state, reports[0].in_flight) == (DrainState.DRAINED, 0)


# ---------------------------------------------------------------------------
# Middleware: admission gate and the finally-release contract
# ---------------------------------------------------------------------------


def _http_scope(method="POST", path="/inspect"):
    return {"type": "http", "method": method, "path": path, "headers": []}


async def _call(middleware, scope):
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    await middleware(scope, receive, send)
    return messages


def _response(messages):
    start = next(m for m in messages if m["type"] == "http.response.start")
    raw = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return start["status"], json.loads(raw)


def test_middleware_counts_during_and_releases_after_the_request():
    coordinator = DrainCoordinator()
    seen_in_flight = []

    async def inner(scope, receive, send):
        seen_in_flight.append(coordinator.in_flight)
        await JSONResponse({"ok": True})(scope, receive, send)

    middleware = InspectAdmissionMiddleware(inner, coordinator)
    status_code, _ = _response(asyncio.run(_call(middleware, _http_scope())))
    assert status_code == 200
    assert seen_in_flight == [1]
    assert coordinator.in_flight == 0


def test_middleware_leaves_other_routes_alone_even_when_drained():
    coordinator = DrainCoordinator()
    asyncio.run(coordinator.drain())
    reached = []

    async def inner(scope, receive, send):
        reached.append((scope["method"], scope["path"]))
        await JSONResponse({"status": "ok"})(scope, receive, send)

    middleware = InspectAdmissionMiddleware(inner, coordinator)
    for method, path in (("GET", "/healthz"), ("POST", "/drain"), ("GET", "/inspect")):
        status_code, _ = _response(asyncio.run(_call(middleware, _http_scope(method, path))))
        assert status_code == 200
    assert reached == [("GET", "/healthz"), ("POST", "/drain"), ("GET", "/inspect")]


def test_middleware_refuses_inspect_with_structured_503():
    coordinator = DrainCoordinator()
    inner_called = False

    async def inner(scope, receive, send):
        nonlocal inner_called
        inner_called = True

    middleware = InspectAdmissionMiddleware(inner, coordinator)

    async def while_draining():
        coordinator.try_admit()
        drain_task = asyncio.create_task(coordinator.drain())
        await asyncio.sleep(0.02)
        try:
            return await _call(middleware, _http_scope())
        finally:
            coordinator.release()
            await drain_task

    status_code, body = _response(asyncio.run(while_draining()))
    assert status_code == 503
    assert body["detail"]["code"] == "SERVICE_DRAINING"
    assert body["detail"]["state"] == "DRAINING"
    assert not inner_called

    status_code, body = _response(asyncio.run(_call(middleware, _http_scope())))
    assert status_code == 503
    assert body["detail"]["code"] == "SERVICE_DRAINING"
    assert body["detail"]["state"] == "DRAINED"


def test_middleware_releases_the_slot_on_internal_error():
    coordinator = DrainCoordinator()

    async def broken(scope, receive, send):
        raise RuntimeError("boom")

    middleware = InspectAdmissionMiddleware(broken, coordinator)
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(_call(middleware, _http_scope()))
    assert coordinator.in_flight == 0


def test_middleware_releases_the_slot_on_cancellation():
    coordinator = DrainCoordinator()

    async def cancelled(scope, receive, send):
        raise asyncio.CancelledError()

    middleware = InspectAdmissionMiddleware(cancelled, coordinator)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_call(middleware, _http_scope()))
    assert coordinator.in_flight == 0
    # The cancelled request must not keep a later drain waiting.
    report = asyncio.run(coordinator.drain())
    assert report.state is DrainState.DRAINED


# ---------------------------------------------------------------------------
# End to end through the real app
# ---------------------------------------------------------------------------


def test_requests_flow_normally_until_drain_is_called():
    coordinator, app = _build()
    client = TestClient(app)
    response = client.post("/inspect", json=PASS_PAYLOAD)
    assert response.status_code == 200
    assert response.json() == {"result": "PASS"}
    assert client.post("/inspect", json=INVALID_PAYLOAD).status_code == 422
    assert client.get("/healthz").json() == {"status": "ok"}
    assert coordinator.state is DrainState.ACCEPTING
    assert coordinator.in_flight == 0


def test_drain_waits_for_inflight_and_rejects_newcomers(monkeypatch):
    coordinator, app = _build()
    entered = threading.Event()
    gate = threading.Event()

    def gated_inspect(grid, labels):
        entered.set()
        assert gate.wait(timeout=10), "test never released the inspection"
        return real_inspect(grid, labels)

    monkeypatch.setattr(main_module, "inspect", gated_inspect)

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=3) as pool:
        try:
            inspect_future = pool.submit(client.post, "/inspect", json=PASS_PAYLOAD)
            assert entered.wait(timeout=5), "inspection never started"

            drain_future = pool.submit(client.post, "/drain")
            assert _eventually(lambda: coordinator.state is DrainState.DRAINING)
            time.sleep(0.2)
            assert not drain_future.done(), "drain finished before the in-flight inspection"

            # New inspections are refused with a structured 503; health stays up.
            rejected = client.post("/inspect", json=PASS_PAYLOAD)
            assert rejected.status_code == 503
            detail = rejected.json()["detail"]
            assert detail["code"] == "SERVICE_DRAINING"
            assert detail["state"] == "DRAINING"
            assert client.get("/healthz").json() == {"status": "ok"}
        finally:
            gate.set()

        # The pre-drain inspection completes with its original verdict...
        response = inspect_future.result(timeout=5)
        assert response.status_code == 200
        assert response.json() == {"result": "PASS"}
        # ...and only then does the drain report DRAINED.
        drained = drain_future.result(timeout=5)
        assert drained.status_code == 200
        assert drained.json() == DRAINED_BODY

        # DRAINED is terminal: refusals now report the drained state.
        again = client.post("/inspect", json=PASS_PAYLOAD)
        assert again.status_code == 503
        assert again.json()["detail"]["state"] == "DRAINED"


def test_inflight_validation_rejection_completes_during_drain(monkeypatch):
    coordinator, app = _build()
    entered = threading.Event()
    gate = threading.Event()
    real_validate = main_module.validate_payload

    def gated_validate(grid, labels):
        entered.set()
        assert gate.wait(timeout=10), "test never released the validation"
        return real_validate(grid, labels)

    monkeypatch.setattr(main_module, "validate_payload", gated_validate)

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as pool:
        try:
            inspect_future = pool.submit(client.post, "/inspect", json=INVALID_PAYLOAD)
            assert entered.wait(timeout=5), "validation never started"
            drain_future = pool.submit(client.post, "/drain")
            assert _eventually(lambda: coordinator.state is DrainState.DRAINING)
        finally:
            gate.set()

        # The 422 rejection admitted before the drain still completes...
        response = inspect_future.result(timeout=5)
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "INPUT_REJECTED"
        # ...and it still releases the drain waiter.
        assert drain_future.result(timeout=5).json() == DRAINED_BODY


def test_internal_error_releases_the_inflight_slot(monkeypatch):
    coordinator, app = _build()

    def broken_inspect(grid, labels):
        raise RuntimeError("inspector exploded")

    monkeypatch.setattr(main_module, "inspect", broken_inspect)
    client = TestClient(app)
    with pytest.raises(RuntimeError, match="inspector exploded"):
        client.post("/inspect", json=PASS_PAYLOAD)
    assert coordinator.in_flight == 0
    # A drain started now must not wait for the crashed request.
    response = client.post("/drain")
    assert response.status_code == 200
    assert response.json() == DRAINED_BODY


def test_concurrent_drain_calls_return_the_same_report(monkeypatch):
    coordinator, app = _build()
    entered = threading.Event()
    gate = threading.Event()

    def gated_inspect(grid, labels):
        entered.set()
        assert gate.wait(timeout=10), "test never released the inspection"
        return real_inspect(grid, labels)

    monkeypatch.setattr(main_module, "inspect", gated_inspect)

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=5) as pool:
        try:
            inspect_future = pool.submit(client.post, "/inspect", json=PASS_PAYLOAD)
            assert entered.wait(timeout=5), "inspection never started"
            drain_futures = [pool.submit(client.post, "/drain") for _ in range(4)]
            assert _eventually(lambda: coordinator.state is DrainState.DRAINING)
        finally:
            gate.set()

        assert inspect_future.result(timeout=5).status_code == 200
        bodies = [future.result(timeout=5).json() for future in drain_futures]
        assert bodies == [DRAINED_BODY] * 4
        # A latecomer drain reuses the finished transition.
        assert client.post("/drain").json() == DRAINED_BODY


def test_only_a_restart_restores_admission():
    _, app = _build()
    client = TestClient(app)
    assert client.post("/drain").json() == DRAINED_BODY
    refused = client.post("/inspect", json=PASS_PAYLOAD)
    assert refused.status_code == 503
    assert refused.json()["detail"]["state"] == "DRAINED"

    # A process restart means a fresh coordinator and app...
    _, restarted_app = _build()
    restarted = TestClient(restarted_app)
    response = restarted.post("/inspect", json=PASS_PAYLOAD)
    assert response.status_code == 200
    assert response.json() == {"result": "PASS"}
