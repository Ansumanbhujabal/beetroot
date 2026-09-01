"""Inverted tag index over the recipe catalog, built for O(constraints) filtering."""

from collections.abc import Iterable, Iterator

from beatroot.trusted.catalog import Recipe


class TagIndex:
    """Inverted index: tag -> bitmap over recipe positions.

    Constraint filtering is set intersection, not iteration. Excluding a tag is
    one bitwise AND-NOT; counting survivors is a popcount. Tag constraints cost
    nothing per catalog item — one AND-NOT per excluded tag, regardless of how
    many recipes exist. Walking the survivors afterward (`iter_ids`) costs
    O(popcount) — one iteration per surviving bit, via `mask & -mask` to
    isolate the lowest set bit — not O(catalog), so filtering a huge catalog
    down to a small surviving set stays cheap end to end. Spec §17.
    """

    def __init__(self, recipes: list[Recipe]) -> None:
        self._ids = [r.id for r in recipes]
        self._pos = {rid: i for i, rid in enumerate(self._ids)}
        self._by_id = {r.id: r for r in recipes}
        self._all = (1 << len(self._ids)) - 1
        self._by_tag: dict[str, int] = {}
        for i, recipe in enumerate(recipes):
            bit = 1 << i
            for tag in recipe.tags:
                self._by_tag[tag] = self._by_tag.get(tag, 0) | bit

    def all_mask(self) -> int:
        return self._all

    def tag_mask(self, tag: str) -> int:
        return self._by_tag.get(tag, 0)

    def exclude_tags(self, mask: int, tags: Iterable[str]) -> int:
        for tag in tags:
            mask &= ~self._by_tag.get(tag, 0)
        return mask & self._all

    def require_tags(self, mask: int, tags: Iterable[str]) -> int:
        for tag in tags:
            mask &= self._by_tag.get(tag, 0)
        return mask & self._all

    def mask_for_ids(self, ids: Iterable[str]) -> int:
        mask = 0
        for rid in ids:
            if rid in self._pos:
                mask |= 1 << self._pos[rid]
        return mask

    def count(self, mask: int) -> int:
        return mask.bit_count()

    def iter_ids(self, mask: int) -> Iterator[str]:
        """Yield only the ids whose bits are set — O(popcount), not O(catalog).

        `mask & -mask` isolates the lowest set bit; `bit_length() - 1` turns it
        into a position. Walking set bits directly, instead of testing every
        position with `mask >> i & 1`, is what makes the surviving-id walk
        cost proportional to the number of survivors rather than the size of
        the catalog.
        """
        while mask:
            low = mask & -mask
            yield self._ids[low.bit_length() - 1]
            mask ^= low

    def to_ids(self, mask: int) -> list[str]:
        return list(self.iter_ids(mask))

    def recipe(self, recipe_id: str) -> Recipe | None:
        """O(1) lookup from a surviving id back to its recipe object."""
        return self._by_id.get(recipe_id)
