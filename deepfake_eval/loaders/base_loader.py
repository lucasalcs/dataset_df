import pandas as pd
from typing import Dict, Union
from ..metrics import (
    compute_eer as _compute_eer,
    compute_auc as _compute_auc,
    compute_metrics_at as _compute_metrics_at,
    find_threshold_for_far as _find_threshold_for_far,
    compute_det_curve as _compute_det_curve,
)

class BaseLoader:
    """
    Base class for all loaders.

    Subclasses *must* set:
        dataset_name: str

    Constructor accepts:
        df: pandas.DataFrame with columns ['relative_path','subset','df_class','score']
        model_name: str, e.g. 'AASIST'

    Provides:
      - self.model_name
      - self.dataset_name
      - to_dataframe() → DataFrame with extra 'dataset' and 'model' cols
      - __len__ / __getitem__ returning a dict with keys
         ['dataset','model','relative_path','subset','df_class','score']
    """
    dataset_name: str  # must be overridden in subclass

    def __init__(self, df: pd.DataFrame, model_name: str):
        if not getattr(self, 'dataset_name', None):
            raise ValueError("Subclasses of BaseLoader must set `dataset_name`")
        self.model_name = model_name

        # Validate required columns
        required = ['relative_path', 'subset', 'df_class', 'score']
        optional = ['tts']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Keep required + existing optional columns
        keep_cols = required + [c for c in optional if c in df.columns]
        self.df = df[keep_cols].copy().reset_index(drop=True)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the DataFrame with 'dataset' and 'model' prepended."""
        out = self.df.copy()
        out.insert(0, 'dataset', self.dataset_name)
        out.insert(1, 'model',   self.model_name)
        return out

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, float]]:
        row = self.df.iloc[idx]
        return {
            'dataset':       self.dataset_name,
            'model':         self.model_name,
            'relative_path': row['relative_path'],
            'subset':        row['subset'],
            'df_class':      row['df_class'],
            'score':         float(row['score']),
        }

    def eer(
        self,
        pos_label: str = "spoof",
        score_col: str = "score",
        label_col: str = "df_class"
    ) -> tuple[float, float]:
        """
        Compute Equal Error Rate (EER) and threshold for this dataset.

        Returns:
            (eer, threshold)
        """
        df = self.to_dataframe()
        return _compute_eer(df, pos_label=pos_label, score_col=score_col, label_col=label_col)

    def auc(
        self,
        pos_label: str = "spoof",
        score_col: str = "score",
        label_col: str = "df_class"
    ) -> float:
        """
        Compute ROC AUC for this dataset.
        """
        df = self.to_dataframe()
        return _compute_auc(df, pos_label=pos_label, score_col=score_col, label_col=label_col)

    def metrics_at(
        self,
        threshold: float,
        pos_label: str = "spoof",
        score_col: str = "score",
        label_col: str = "df_class"
    ) -> dict:
        """
        Compute classification metrics at a specified threshold.
        """
        df = self.to_dataframe()
        return _compute_metrics_at(df, threshold, pos_label=pos_label, score_col=score_col, label_col=label_col)

    def threshold_for_far(
        self,
        target_far: float,
        pos_label: str = "spoof",
        score_col: str = "score",
        label_col: str = "df_class"
    ) -> float:
        """
        Find threshold corresponding to a target False Alarm Rate (FAR).
        """
        df = self.to_dataframe()
        return _find_threshold_for_far(df, target_far, pos_label=pos_label, score_col=score_col, label_col=label_col)

    def det_curve(
        self,
        pos_label: str = "spoof",
        score_col: str = "score",
        label_col: str = "df_class"
    ) -> tuple[list[float], list[float], list[float]]:
        """
        Compute Detection Error Trade-off (DET) curve.

        Returns:
            (false rejection rates, false acceptance rates, thresholds)
        """
        df = self.to_dataframe()
        target_scores = df[df[label_col] != pos_label][score_col].to_numpy()
        nontarget_scores = df[df[label_col] == pos_label][score_col].to_numpy()
        return _compute_det_curve(target_scores, nontarget_scores) 