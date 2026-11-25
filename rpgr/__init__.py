"""RPGR: Reciprocal Prototype-Guided Re-ranking for WSI-report cross-modal retrieval."""
from .calibration import direct_term, reciprocal_term
from .prototypes import AdaptiveSettings, build_adaptive_prototypes_for_case
from .rerank import rank_gallery, rpgr_score, rpgr_scores_both_directions
from .similarity import global_similarity, l2_normalize_rows, mean_pool, prototype_similarity

__all__ = [
    "AdaptiveSettings", "build_adaptive_prototypes_for_case", "direct_term", "global_similarity",
    "l2_normalize_rows", "mean_pool", "prototype_similarity", "rank_gallery", "reciprocal_term",
    "rpgr_score", "rpgr_scores_both_directions",
]
__version__ = "1.0.0"
