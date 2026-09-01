"""Tests for `POST /feedback`. Spec §9, §15.

Uses the same module-scoped `test_container`/`client` fixtures as
`test_routes.py` (see `tests/api/conftest.py`).
"""


def _a_real_recipe_id(test_container) -> str:
    return test_container.catalog.recipes()[0].id


def test_feedback_returns_the_updated_affinity(client, test_container):
    recipe_id = _a_real_recipe_id(test_container)
    recipe = test_container.catalog.recipe(recipe_id)

    r = client.post(
        "/feedback",
        json={"profile_id": "fb-p1", "recipe_id": recipe_id, "accepted": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    for tag in recipe.tags:
        assert body["affinity"][tag] > 0


def test_feedback_rejection_lowers_affinity(client, test_container):
    recipe_id = _a_real_recipe_id(test_container)
    recipe = test_container.catalog.recipe(recipe_id)

    r = client.post(
        "/feedback",
        json={"profile_id": "fb-p2", "recipe_id": recipe_id, "accepted": False},
    )
    assert r.status_code == 200
    body = r.json()
    for tag in recipe.tags:
        assert body["affinity"][tag] < 0


def test_feedback_unknown_recipe_returns_404(client):
    r = client.post(
        "/feedback",
        json={"profile_id": "fb-p3", "recipe_id": "does-not-exist", "accepted": True},
    )
    assert r.status_code == 404


def test_feedback_affinity_visible_via_container_directly(client, test_container):
    """The response body isn't the only place the effect must show up —
    the same profile's affinity read straight off the Container's
    PreferenceMemory must agree."""
    recipe_id = _a_real_recipe_id(test_container)
    recipe = test_container.catalog.recipe(recipe_id)

    client.post(
        "/feedback",
        json={"profile_id": "fb-p4", "recipe_id": recipe_id, "accepted": True},
    )
    stored = test_container.preferences.affinity("fb-p4")
    for tag in recipe.tags:
        assert stored[tag] > 0
