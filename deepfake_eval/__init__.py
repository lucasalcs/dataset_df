from .metrics import (
    compute_metrics,
    compute_eer,
    compute_det_curve,
    compute_auc,
    compute_metrics_at,
    find_threshold_for_far,
)
from .utils import check_key_match
from .plotting import (
    set_style,
    plot_score_histogram,
    plot_kde_by_tts,
    plot_class_distribution,
    plot_rate_by_group,
    plot_confusion_matrix,
)
from .loaders import (
    ASVspoof2019LALoader,
    ASVspoof2021DFLoader,
    BRSpeechDFLoader,
)

# Define __all__ for explicit public API
__all__ = [
    # Loader classes
    'ASVspoof2019LALoader',
    'ASVspoof2021DFLoader',
    'BRSpeechDFLoader',
    # Core metrics
    'compute_metrics',
    'compute_eer',
    'compute_det_curve',
    'compute_auc',
    'compute_metrics_at',
    'find_threshold_for_far',
    # Utilities
    'check_key_match',
    # Plotting functions
    'set_style',
    'plot_score_histogram',
    'plot_kde_by_tts',
    'plot_class_distribution',
    'plot_rate_by_group',
    'plot_confusion_matrix',
]

# Optional: Add docstring for the package
"""
DeepFake Eval Package

Provides tools for loading deepfake detection scores (from pickles and metadata)
and computing evaluation metrics like EER, AUC, etc.

Key Components:
- Loaders (in deepfake_eval.loaders): Classes like ASVspoof2019LALoader, ASVspoof2021DFLoader, BRSpeechDFLoader.
- Metrics (in deepfake_eval.metrics): Functions like compute_eer, compute_auc, compute_metrics_at, find_threshold_for_far.
- Utils (in deepfake_eval.utils): Helper functions (e.g., check_key_match).
- Plotting (in deepfake_eval.plotting): Visualization helpers.
"""
