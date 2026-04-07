"""Reciprocal Rank Fusion for combining multiple ranked result lists."""

from typing import List, Optional, Tuple
from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
    weights: Optional[List[float]] = None,
    top_k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for document d = sum over lists L of: weight_L / (k + rank_of_d_in_L)
    Documents not in a list are skipped for that list (no penalty).

    Args:
        ranked_lists: Each list is [(path, score)] sorted descending by score.
        k: Ranking constant. Higher k reduces gap between ranks. Standard: 60.
        weights: Optional weight per list. Defaults to equal weights (1.0).
        top_k: Optional limit on returned results.

    Returns:
        Fused [(path, rrf_score)] sorted descending.
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    rrf_scores: dict[str, float] = defaultdict(float)

    for lst, weight in zip(ranked_lists, weights):
        for rank, (path, _score) in enumerate(lst, start=1):
            rrf_scores[path] += weight / (k + rank)

    results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    if top_k is not None:
        results = results[:top_k]

    return results
