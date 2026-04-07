"""Tests for Reciprocal Rank Fusion (engines/rrf.py)."""

import sys
from pathlib import Path

import pytest

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from engines.rrf import reciprocal_rank_fusion


class TestBasicFusion:
    """Two ranked lists, document ranked high in both should win."""

    def test_top_result_ranked_high_in_both(self):
        list_a = [("doc1", 10.0), ("doc2", 8.0), ("doc3", 5.0)]
        list_b = [("doc1", 0.9), ("doc3", 0.7), ("doc2", 0.3)]
        result = reciprocal_rank_fusion([list_a, list_b])
        assert result[0][0] == "doc1"

    def test_fusion_returns_all_documents(self):
        list_a = [("doc1", 10.0), ("doc2", 8.0)]
        list_b = [("doc1", 0.9), ("doc3", 0.7)]
        result = reciprocal_rank_fusion([list_a, list_b])
        paths = [r[0] for r in result]
        assert set(paths) == {"doc1", "doc2", "doc3"}


class TestDisjointLists:
    """Documents appearing in only one list still get scored."""

    def test_disjoint_documents_scored(self):
        list_a = [("doc1", 10.0), ("doc2", 8.0)]
        list_b = [("doc3", 0.9), ("doc4", 0.7)]
        result = reciprocal_rank_fusion([list_a, list_b])
        paths = [r[0] for r in result]
        assert set(paths) == {"doc1", "doc2", "doc3", "doc4"}

    def test_disjoint_scores_positive(self):
        list_a = [("doc1", 10.0)]
        list_b = [("doc2", 0.9)]
        result = reciprocal_rank_fusion([list_a, list_b])
        for _path, score in result:
            assert score > 0


class TestSingleList:
    """Single list preserves rank order."""

    def test_single_list_order_preserved(self):
        lst = [("doc1", 10.0), ("doc2", 8.0), ("doc3", 5.0)]
        result = reciprocal_rank_fusion([lst])
        assert [r[0] for r in result] == ["doc1", "doc2", "doc3"]


class TestEmptyLists:
    """Edge cases with empty inputs."""

    def test_no_lists(self):
        assert reciprocal_rank_fusion([]) == []

    def test_single_empty_list(self):
        assert reciprocal_rank_fusion([[]]) == []

    def test_mix_empty_and_nonempty(self):
        result = reciprocal_rank_fusion([[], [("doc1", 1.0)]])
        assert len(result) == 1
        assert result[0][0] == "doc1"


class TestWeightedLists:
    """Higher weight makes that list's top document win."""

    def test_high_weight_shifts_winner(self):
        # doc1 is top in list_a, doc2 is top in list_b
        list_a = [("doc1", 10.0), ("doc2", 5.0)]
        list_b = [("doc2", 10.0), ("doc1", 5.0)]
        # Equal weights: tied (both appear rank 1 and rank 2)
        # Heavy weight on list_b: doc2 should win
        result = reciprocal_rank_fusion([list_a, list_b], weights=[0.1, 10.0])
        assert result[0][0] == "doc2"

    def test_equal_weights_same_as_default(self):
        lists = [[("doc1", 10.0), ("doc2", 5.0)]]
        r1 = reciprocal_rank_fusion(lists, weights=[1.0])
        r2 = reciprocal_rank_fusion(lists)
        assert r1 == r2


class TestKParameter:
    """Lower k gives more weight to top ranks (larger gap between rank 1 and 2)."""

    def test_lower_k_amplifies_top_rank(self):
        lst = [("doc1", 10.0), ("doc2", 8.0)]
        # With k=1: rank1 gets 1/(1+1)=0.5, rank2 gets 1/(1+2)=0.333
        # Gap = 0.167
        result_low_k = reciprocal_rank_fusion([lst], k=1)
        # With k=100: rank1 gets 1/101=0.0099, rank2 gets 1/102=0.0098
        # Gap = 0.0001
        result_high_k = reciprocal_rank_fusion([lst], k=100)

        gap_low = result_low_k[0][1] - result_low_k[1][1]
        gap_high = result_high_k[0][1] - result_high_k[1][1]
        assert gap_low > gap_high


class TestTopKLimit:
    """top_k limits the number of returned results."""

    def test_top_k_limits_output(self):
        lst = [("doc1", 10.0), ("doc2", 8.0), ("doc3", 5.0)]
        result = reciprocal_rank_fusion([lst], top_k=2)
        assert len(result) == 2

    def test_top_k_none_returns_all(self):
        lst = [("doc1", 10.0), ("doc2", 8.0), ("doc3", 5.0)]
        result = reciprocal_rank_fusion([lst], top_k=None)
        assert len(result) == 3

    def test_top_k_larger_than_results(self):
        lst = [("doc1", 10.0)]
        result = reciprocal_rank_fusion([lst], top_k=10)
        assert len(result) == 1


class TestThreeLists:
    """Works with more than two lists."""

    def test_three_lists_all_docs_present(self):
        l1 = [("a", 1.0), ("b", 0.5)]
        l2 = [("b", 1.0), ("c", 0.5)]
        l3 = [("c", 1.0), ("a", 0.5)]
        result = reciprocal_rank_fusion([l1, l2, l3])
        paths = {r[0] for r in result}
        assert paths == {"a", "b", "c"}

    def test_three_lists_symmetric_scores(self):
        # Each doc is rank 1 once and rank 2 once, so all should tie
        l1 = [("a", 1.0), ("b", 0.5)]
        l2 = [("b", 1.0), ("c", 0.5)]
        l3 = [("c", 1.0), ("a", 0.5)]
        result = reciprocal_rank_fusion([l1, l2, l3])
        scores = [r[1] for r in result]
        assert scores[0] == pytest.approx(scores[1])
        assert scores[1] == pytest.approx(scores[2])


class TestDuplicatesWithinList:
    """Duplicate paths within a single list sum contributions."""

    def test_duplicate_path_sums(self):
        lst = [("doc1", 10.0), ("doc1", 5.0)]
        result = reciprocal_rank_fusion([lst])
        # doc1 appears at rank 1 and rank 2: 1/(60+1) + 1/(60+2)
        doc1_entries = [r for r in result if r[0] == "doc1"]
        assert len(doc1_entries) == 1
        expected = 1.0 / 61 + 1.0 / 62
        assert doc1_entries[0][1] == pytest.approx(expected)


class TestScoreValues:
    """Verify RRF scores match the 1/(k+rank) formula exactly."""

    def test_single_list_scores(self):
        lst = [("doc1", 99.0), ("doc2", 50.0), ("doc3", 1.0)]
        result = reciprocal_rank_fusion([lst], k=60)
        scores = {r[0]: r[1] for r in result}
        assert scores["doc1"] == pytest.approx(1.0 / 61)
        assert scores["doc2"] == pytest.approx(1.0 / 62)
        assert scores["doc3"] == pytest.approx(1.0 / 63)

    def test_two_list_scores(self):
        l1 = [("a", 10.0), ("b", 5.0)]
        l2 = [("b", 10.0), ("a", 5.0)]
        result = reciprocal_rank_fusion([l1, l2], k=60)
        scores = {r[0]: r[1] for r in result}
        # a: rank1 in l1 + rank2 in l2 = 1/61 + 1/62
        # b: rank2 in l1 + rank1 in l2 = 1/62 + 1/61
        expected = 1.0 / 61 + 1.0 / 62
        assert scores["a"] == pytest.approx(expected)
        assert scores["b"] == pytest.approx(expected)

    def test_weighted_scores(self):
        lst = [("doc1", 10.0)]
        result = reciprocal_rank_fusion([lst], k=60, weights=[3.0])
        assert result[0][1] == pytest.approx(3.0 / 61)
