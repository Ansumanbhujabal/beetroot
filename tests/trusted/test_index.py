from beatroot.trusted.index import TagIndex


class _R:
    def __init__(self, rid, tags, prep=20):
        self.id, self.tags, self.prep_minutes = rid, set(tags), prep
        self.nutrition = self.cost_inr = None


RECIPES = [
    _R("a", {"vegan", "peanut"}),
    _R("b", {"vegan", "dairy"}),
    _R("c", {"vegetarian", "dairy"}),
    _R("d", {"vegan"}, prep=90),
]


def test_excluding_a_tag_is_a_bitmask_operation():
    idx = TagIndex(RECIPES)
    survivors = idx.exclude_tags(idx.all_mask(), ["peanut"])
    assert idx.to_ids(survivors) == ["b", "c", "d"]


def test_popcount_gives_survivor_count_without_materialising_ids():
    idx = TagIndex(RECIPES)
    assert idx.count(idx.exclude_tags(idx.all_mask(), ["dairy"])) == 2


def test_excluding_an_unknown_tag_removes_nothing():
    idx = TagIndex(RECIPES)
    assert idx.count(idx.exclude_tags(idx.all_mask(), ["unicorn"])) == 4


def test_index_is_independent_of_catalog_size_in_operations():
    """The whole point: one AND per constraint, regardless of catalog size."""
    big = [_R(f"r{i}", {"vegan"} if i % 2 else {"peanut"}) for i in range(10_000)]
    idx = TagIndex(big)
    mask = idx.exclude_tags(idx.all_mask(), ["peanut"])
    assert idx.count(mask) == 5000


def test_ids_round_trip_through_the_mask():
    idx = TagIndex(RECIPES)
    assert set(idx.to_ids(idx.mask_for_ids(["a", "c"]))) == {"a", "c"}
