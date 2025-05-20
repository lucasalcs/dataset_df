import numpy as np
from typing import Tuple, Dict, Optional, Union, List
import pandas as pd
from pathlib import Path
import sys
import os.path
import pickle
from sklearn.metrics import roc_curve, roc_auc_score

def compute_det_curve(target_scores: np.ndarray, nontarget_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Detection Error Trade-off (DET) curve.

    Args:
        target_scores: Array of scores for target (bona fide) samples
        nontarget_scores: Array of scores for non-target (deepfake) samples

    Returns:
        Tuple containing (false rejection rates, false acceptance rates, thresholds)
    """
    # Handle empty inputs gracefully
    if len(target_scores) == 0 and len(nontarget_scores) == 0:
        return np.array([0.0]), np.array([1.0]), np.array([-np.inf]) # Default DET

    target_scores = np.asarray(target_scores).astype(float).ravel()
    nontarget_scores = np.asarray(nontarget_scores).astype(float).ravel()

    n_tar = len(target_scores)
    n_non = len(nontarget_scores)

    if n_tar == 0:
        # Only nontargets: FRR is always 1, FAR drops from 1 to 0
        scores = np.sort(nontarget_scores)
        thresholds = np.concatenate(([scores[0] - 1e-6], scores))
        frr = np.ones_like(thresholds)
        far = np.linspace(1, 0, len(thresholds))
        return frr, far, thresholds

    if n_non == 0:
        # Only targets: FAR is always 0, FRR drops from 1 to 0
        scores = np.sort(target_scores)
        thresholds = np.concatenate(([scores[0] - 1e-6], scores))
        far = np.zeros_like(thresholds)
        frr = np.linspace(1, 0, len(thresholds))
        return frr, far, thresholds

    # Standard case with both targets and nontargets
    n_scores = n_tar + n_non
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(n_tar), np.zeros(n_non)))

    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]
    scores_sorted = all_scores[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = n_non - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / n_tar)) 
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / n_non))
    thresholds = np.concatenate((np.atleast_1d(scores_sorted[0] - 1e-6), scores_sorted))

    # Remove redundant points where rates don't change
    # Keep the last point for each unique threshold
    _, unique_indices = np.unique(thresholds, return_index=True)
    # Ensure indices are sorted if np.unique doesn't guarantee it (it should)
    unique_indices = np.sort(unique_indices)

    # Add the first point (0,1) explicitly if not already covered
    # And the last point (potentially 1,0)
    final_indices = np.unique(np.concatenate(([0], unique_indices, [len(frr)-1])))

    return frr[final_indices], far[final_indices], thresholds[final_indices]

def compute_eer(
    df: pd.DataFrame,
    pos_label: str = "spoof",
    score_col: str = "score",
    label_col: str = "df_class"
) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and the corresponding threshold from a DataFrame.

    Finds the threshold where FRR is closest to FAR.
    Assumes higher scores => bona-fide (negative class).

    Args:
        df (pd.DataFrame): DataFrame containing scores and labels.
        pos_label (str): The label value representing the positive (spoof) class.
        score_col (str): The name of the column containing scores.
        label_col (str): The name of the column containing labels.

    Returns:
        Tuple containing (EER, threshold).
        Returns (NaN, NaN) if either class has zero samples.
    """
    # Extract target (bona fide = negative) and non-target (spoof = positive) scores
    # Note: The internal compute_det_curve expects target_scores (bona fide) first
    target_scores = df[df[label_col] != pos_label][score_col].to_numpy()
    nontarget_scores = df[df[label_col] == pos_label][score_col].to_numpy()

    if len(target_scores) == 0 or len(nontarget_scores) == 0:
        print("Warning: Cannot compute EER with zero samples in one or both classes.")
        return np.nan, np.nan

    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)

    # Find the point where abs(FRR - FAR) is minimal
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)

    # EER is the average of FRR and FAR at this point
    eer = (frr[min_index] + far[min_index]) / 2.0
    eer_threshold = thresholds[min_index]

    return eer, eer_threshold

def compute_metrics(target_scores: np.ndarray,
                   nontarget_scores: np.ndarray) -> Dict:
    """
    Compute standard evaluation metrics (EER, threshold, counts).

    Args:
        target_scores: Array of scores for target (bona fide) samples.
        nontarget_scores: Array of scores for non-target (deepfake) samples.

    Returns:
        Dictionary containing computed metrics:
          {'eer': float, 'eer_threshold': float, 'num_target': int, 'num_nontarget': int}
        EER and threshold will be NaN if computation fails (e.g., empty inputs).
    """
    # Ensure inputs are numpy arrays
    target_scores = np.asarray(target_scores).ravel()
    nontarget_scores = np.asarray(nontarget_scores).ravel()

    eer, eer_threshold = compute_eer(target_scores, nontarget_scores)

    return {
        'eer': eer,
        'eer_threshold': eer_threshold,
        'num_target': len(target_scores),
        'num_nontarget': len(nontarget_scores)
    }

# --- Functions migrated from df_evaluation/metrics.py --- #

def compute_auc(
    df: pd.DataFrame,
    pos_label: str = "spoof",
    score_col: str = "score",
    label_col: str = "df_class"
) -> float:
    """
    Compute ROC AUC, assuming higher scores => bona-fide.
    Internally inverts scores to match roc_auc_score convention.
    Requires DataFrame input.
    """
    y_true = (df[label_col] == pos_label).astype(int).to_numpy()
    scores = df[score_col].to_numpy()
    # Invert scores because roc_auc_score expects higher = positive class (spoof)
    return roc_auc_score(y_true, -scores)

def get_confusion(
    df: pd.DataFrame,
    threshold: float,
    pos_label: str = "spoof",
    score_col: str = "score",
    label_col: str = "df_class"
) -> Dict[str, int]:
    """
    Compute confusion counts at a given threshold, where:
      - score < threshold ⇒ predict spoof (positive)
      - score ≥ threshold ⇒ predict bona-fide (negative)
    Requires DataFrame input.
    Returns dict with keys: tp, fp, tn, fn.
    """
    y_true = (df[label_col] == pos_label).astype(int).to_numpy()
    scores = df[score_col].to_numpy()
    # Prediction: 1 if score < threshold (predict spoof), 0 otherwise
    y_pred = (scores < threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

def compute_metrics_at(
    df: pd.DataFrame,
    threshold: float,
    pos_label: str = "spoof",
    score_col: str = "score",
    label_col: str = "df_class"
) -> Dict[str, float]:
    """
    Compute various metrics at a specified threshold:
      - tpr, fpr, fnr, precision, recall, f1, accuracy
    Requires DataFrame input.
    """
    c = get_confusion(df, threshold, pos_label, score_col, label_col)
    tp, fp, tn, fn = c['tp'], c['fp'], c['tn'], c['fn']
    total = tp + tn + fp + fn

    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tpr
    accuracy = (tp + tn) / total if total else 0.0
    f1 = 2 * tp / (2*tp + fp + fn) if (2*tp + fp + fn) else 0.0

    return {
        "threshold": threshold,
        "tpr": tpr,
        "fpr": fpr,
        "fnr": fnr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }

def find_threshold_for_far(
    df: pd.DataFrame,
    target_far: float,
    pos_label: str = "spoof",
    score_col: str = "score",
    label_col: str = "df_class"
) -> float:
    """
    Find the highest threshold such that False Alarm Rate (FPR) ≤ target_far,
    assuming higher scores ⇒ bona-fide.
    Requires DataFrame input.
    FAR = FP / (FP + TN)
    Returns the threshold on the original score scale.
    """
    y_true = (df[label_col] == pos_label).astype(int).to_numpy()
    scores = df[score_col].to_numpy()
    # Invert scores for roc_curve (expects higher = positive class)
    inv_scores = -scores

    fpr, _, thresh_inv = roc_curve(y_true, inv_scores, pos_label=1)
    # Find indices where FPR ≤ target_far
    valid_indices = np.where(fpr <= target_far)[0]
    if valid_indices.size == 0:
        # If no threshold meets target_far, return the threshold corresponding
        # to the lowest FPR (usually the highest inverted threshold)
        # print(f"Warning: No threshold found for target_far={target_far}. Returning threshold for lowest FPR ({np.min(fpr):.4f}).")
        # The last threshold corresponds to predicting everything as negative (bona-fide)
        # It might be -inf, so handle appropriately. Use the first available finite threshold.
        finite_thresh_indices = np.where(np.isfinite(thresh_inv))[0]
        if finite_thresh_indices.size > 0:
             return -thresh_inv[finite_thresh_indices[-1]] # highest finite inverted threshold
        else:
             return np.inf # Or handle as error

    # Pick the threshold corresponding to the lowest FPR that is still <= target_far
    # This corresponds to the largest inverted threshold (least negative/most positive)
    # which is the smallest original threshold (most negative/least positive)
    # Correct: we want the highest *original* threshold, which means the smallest *inverted* threshold
    best_fpr_index = valid_indices[0] # Index of smallest FPR <= target_far
    best_thresh_inv = thresh_inv[best_fpr_index]

    # To get the highest *original* threshold, we need the largest inverted threshold from the valid ones
    best_fpr_index_for_highest_thresh = valid_indices[-1] # Index for highest original threshold
    best_thresh_inv_for_highest_thresh = thresh_inv[best_fpr_index_for_highest_thresh]

    # Return the original threshold
    return -best_thresh_inv_for_highest_thresh


# --- Add migrated functions to __all__ if making public --- #
__all__ = [
    "compute_det_curve",
    "compute_eer",
    "compute_metrics",
    "compute_auc",
    "compute_metrics_at",
    "find_threshold_for_far",
    # "get_confusion", # Usually internal helper
]

# --- evaluate_scores function removed --- #
# --- main function removed --- # 