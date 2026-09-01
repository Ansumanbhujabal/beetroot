from beatroot.retrieval.fusion import rrf


def test_item_ranked_well_by_both_beats_item_ranked_well_by_one():
    lexical = [("a", 0.9), ("b", 0.8), ("c", 0.1)]
    dense = [("b", 0.9), ("a", 0.2), ("c", 0.1)]
    fused = dict(rrf([lexical, dense]))
    assert fused["b"] > fused["c"]
    assert fused["a"] > fused["c"]


def test_weights_shift_the_ordering():
    lexical = [("a", 1.0), ("b", 0.0)]
    dense = [("b", 1.0), ("a", 0.0)]
    lex_heavy = dict(rrf([lexical, dense], weights=[3.0, 1.0]))
    assert lex_heavy["a"] > lex_heavy["b"]
    dense_heavy = dict(rrf([lexical, dense], weights=[1.0, 3.0]))
    assert dense_heavy["b"] > dense_heavy["a"]


def test_missing_from_one_ranking_is_not_fatal():
    fused = dict(rrf([[("a", 1.0)], [("b", 1.0)]]))
    assert set(fused) == {"a", "b"}


def test_empty_input_returns_empty():
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_default_k_comes_from_settings_not_a_literal():
    """No magic `60` in this module — the default k must trace to
    settings.retrieval.rrf_k."""
    from beatroot.settings import get_settings

    k = get_settings().retrieval.rrf_k
    default = dict(rrf([[("a", 1.0)]]))
    explicit = dict(rrf([[("a", 1.0)]], k=k))
    assert default == explicit


def test_rank_position_not_score_drives_fusion():
    """Two rankings with wildly different score scales but the same rank
    order should fuse identically to two rankings that used a normalised
    0..1 scale — RRF only sees position."""
    big_scores = [("a", 987.0), ("b", 12.0), ("c", 0.001)]
    small_scores = [("a", 0.99), ("b", 0.5), ("c", 0.01)]
    assert rrf([big_scores]) == rrf([small_scores])
