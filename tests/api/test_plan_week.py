"""Tests for `POST /plan/week` and `GET /plan/week/{job_id}`.

The two routes are one feature split across a submit and a poll, so these
drive the pair together: submit a week, poll until it lands, read it back.
What is worth pinning at this layer is the handover — that the POST answers
without waiting for the week, and that the id it returns is one the GET can
actually resolve. The planner's own behaviour is covered in
`tests/agent/test_batch_plan.py` and is not re-litigated here.
"""

import time

import pytest

_TIMEOUT_S = 30.0


def _week(profile_id: str, excluded_tag: str, days: int) -> dict:
    """A submission body carrying a real constraint, the way a caller
    reaching this route from the profile picker actually sends one."""
    return {
        "profile_id": profile_id,
        "constraints": [
            {"id": "c1", "kind": "exclude_tag", "severity": "medical", "value": excluded_tag}
        ],
        "query": "something with rice",
        "days": days,
    }


def _poll_until_ready(client, job_id: str) -> dict:
    deadline = time.monotonic() + _TIMEOUT_S
    while True:
        body = client.get(f"/plan/week/{job_id}").json()
        if body["status"] != "running":
            return body
        if time.monotonic() > deadline:
            done = body["completed"]
            pytest.fail(f"week {job_id} still running after {_TIMEOUT_S}s ({done} days done)")
        time.sleep(0.02)


def test_submitting_a_week_returns_202_and_an_id_to_poll(client):
    response = client.post("/plan/week", json=_week("pw1", "peanut", 3))

    assert response.status_code == 202
    body = response.json()
    assert body["job_id"]
    assert body["days_requested"] == 3


def test_the_week_arrives_through_the_poll_route(client):
    job_id = client.post("/plan/week", json=_week("pw2", "shellfish", 4)).json()["job_id"]

    body = _poll_until_ready(client, job_id)

    assert body["status"] == "ready"
    assert body["completed"] == 4
    assert len(body["plan"]["days"]) == 4
    assert body["plan"]["profile_id"] == "pw2"
    assert body["plan"]["totals"]["days_counted"] <= 4


def test_a_day_carries_whichever_terminal_it_reached(client):
    job_id = client.post("/plan/week", json=_week("pw3", "sesame", 2)).json()["job_id"]

    body = _poll_until_ready(client, job_id)

    for day in body["plan"]["days"]:
        assert day["terminal_state"]
        assert (day["recipe_id"] is not None) == (day["terminal_state"] == "COMMIT")


def test_a_malformed_submission_is_rejected_by_the_model(client):
    """`profile_id` carries `RecommendRequest`'s bounds, so an empty one is
    a 422 from the model rather than a job nobody can attribute."""
    assert client.post("/plan/week", json=_week("", "peanut", 3)).status_code == 422
