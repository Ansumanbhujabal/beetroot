"""The /docs page: the architecture diagram and the project's own documents.

Two things here are worth a test rather than a glance.

FastAPI mounts Swagger UI at "/docs" BY DEFAULT, and registers it during
`FastAPI.__init__` — before any route in `api.main`. Adding a page at "/docs"
without moving Swagger would have silently served the OpenAPI explorer
instead, and nothing else in the suite would have noticed.

And `/docs/file/{name}` serves files off disk. It is an allow-list keyed on
the exact filename, not a path joined onto a base directory, so traversal is
not something it defends against — it is something it cannot express. The
tests below hold that property in place, because the tempting "simplification"
is to swap the dict for `DOCS_DIR / name`.
"""

from fastapi.testclient import TestClient

from beatroot.api.main import _DOC_FILES, app

client = TestClient(app)


def test_docs_page_serves_and_is_self_contained() -> None:
    r = client.get("/docs")
    assert r.status_code == 200
    body = r.text
    assert "Architecture &amp; docs" in body or "Architecture & docs" in body
    # Same no-CDN rule the other pages are held to.
    assert "http://" not in body.replace("http://localhost", "")
    assert "https://" not in body


def test_swagger_did_not_shadow_the_docs_page() -> None:
    """If FastAPI's default docs_url ever comes back, /docs silently becomes
    the OpenAPI explorer and this page disappears with no error."""
    assert "swagger" not in client.get("/docs").text.lower()
    assert client.get("/api-docs").status_code == 200


def test_every_advertised_file_is_downloadable() -> None:
    """The page links these by name; a link that 404s is a broken page."""
    for name in _DOC_FILES:
        r = client.get(f"/docs/file/{name}")
        assert r.status_code == 200, f"{name} is advertised but not served"
        assert len(r.content) > 0


def test_page_links_only_to_allow_listed_names() -> None:
    """Guards the other direction: a link on the page that the allow-list does
    not contain would render as a dead button."""
    import re

    body = client.get("/docs").text
    linked = set(re.findall(r'/docs/file/([^"\']+)', body))
    unknown = linked - set(_DOC_FILES)
    assert not unknown, f"page links to files the allow-list does not serve: {unknown}"


def test_unlisted_names_are_404_including_real_files() -> None:
    """`.env` and `settings.py` exist on disk. Neither is reachable, because
    reachability is membership in a dict, not a path that resolves."""
    for name in ("../.env", "../../.env", ".env", "settings.py", "skills-lock.json"):
        assert client.get(f"/docs/file/{name}").status_code == 404, name


def test_allow_list_targets_stay_inside_the_repo() -> None:
    """Structural: every entry must point somewhere under the repo root. A
    future edit adding an absolute path outside it fails here."""
    from beatroot.container import ROOT

    for name, (path, _media) in _DOC_FILES.items():
        assert ROOT in path.resolve().parents, f"{name} points outside the repo"


def test_all_four_pages_share_the_nav() -> None:
    for path in ("/", "/incidents", "/evals", "/docs"):
        body = client.get(path).text
        assert '<nav class="topnav">' in body
        for link in ('href="/"', 'href="/incidents"', 'href="/evals"', 'href="/docs"'):
            assert link in body, f"{path} is missing {link}"
