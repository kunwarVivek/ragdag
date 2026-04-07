"""Cross-encoder reranker for search result refinement."""

from typing import List, Optional, Tuple

_cross_encoder = None


def _get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Lazy-load the cross-encoder model."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(model_name)
    return _cross_encoder


def rerank(
    query: str,
    candidates: List[Tuple[str, float, str]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Rerank candidates using a cross-encoder model.

    Args:
        query: The search query.
        candidates: List of (path, rrf_score, content) tuples.
        model_name: Cross-encoder model name.
        top_k: Optional limit on results.

    Returns:
        List of (path, blended_score) sorted descending.
    """
    if not candidates:
        return []

    try:
        model = _get_cross_encoder(model_name)
    except (ImportError, Exception):
        return [(path, score) for path, score, _ in candidates]

    pairs = [(query, content) for _, _, content in candidates]
    ce_scores = model.predict(pairs)

    # Normalize cross-encoder scores to [0, 1]
    ce_min = min(ce_scores)
    ce_max = max(ce_scores)
    ce_range = ce_max - ce_min if ce_max > ce_min else 1.0
    ce_norm = [(s - ce_min) / ce_range for s in ce_scores]

    # Normalize RRF scores to [0, 1]
    rrf_scores = [score for _, score, _ in candidates]
    rrf_max = max(rrf_scores) if rrf_scores else 1.0
    rrf_norm = [s / rrf_max if rrf_max > 0 else 0 for s in rrf_scores]

    # Blend: RRF 40%, reranker 60%
    results = []
    for i, (path, _orig_score, _content) in enumerate(candidates):
        blended = 0.4 * rrf_norm[i] + 0.6 * ce_norm[i]
        results.append((path, blended))

    results.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        results = results[:top_k]

    return results
