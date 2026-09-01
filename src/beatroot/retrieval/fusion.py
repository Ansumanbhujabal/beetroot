"""Reciprocal Rank Fusion.

RRF combines rankings by POSITION, not by score. BM25, cosine similarity and a
hand-rolled affinity sum live on incomparable scales; trying to normalise and
average them is a losing game (a BM25 score of 8 and a cosine of 0.8 mean
nothing relative to each other). RRF sidesteps the problem entirely: every
ranking contributes `weight / (k + rank + 1)` for each item, so only *where*
an item sits in each list matters, never *how much* it beat its neighbours by.
An item ranked well by several independent signals outranks one ranked well
by only one. Spec §10.
"""

from beatroot.settings import get_settings


def rrf(
    rankings: list[list[tuple[str, float]]],
    k: int | None = None,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists by reciprocal rank, not by score.

    ``rankings`` is a list of per-source rankings, each already sorted best
    first as ``(item_id, score)`` pairs — the incoming score is used only to
    establish that order; RRF discards it in favour of rank position. ``k``
    defaults to ``settings.retrieval.rrf_k`` (no magic number lives here).
    Missing from one ranking is not a penalty beyond simply not contributing
    a term for that source. Ties break on id for a deterministic order.
    """
    if not rankings:
        return []
    if k is None:
        k = get_settings().retrieval.rrf_k
    if weights is None:
        weights = [1.0] * len(rankings)

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for position, (item_id, _score) in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + position + 1)

    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
