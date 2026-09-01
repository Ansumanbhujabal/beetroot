from pathlib import Path

from beatroot.store.db import connect, seed
from beatroot.trusted.catalog import Catalog

DATA = Path(__file__).parents[2] / "data"


def test_catalog_loads_and_derives_tags(tmp_path):
    conn = connect(tmp_path / "t.db")
    seed(conn, DATA)
    cat = Catalog(conn)
    assert len(cat.recipes()) >= 90, "catalog should hold ~100 recipes"
    tagged = [r for r in cat.recipes() if "peanut" in r.tags]
    assert tagged, "at least one recipe must carry a transitively derived peanut tag"


def test_seed_is_idempotent(tmp_path):
    conn = connect(tmp_path / "t.db")
    seed(conn, DATA)
    first = len(Catalog(conn).recipes())
    seed(conn, DATA)
    assert len(Catalog(conn).recipes()) == first
