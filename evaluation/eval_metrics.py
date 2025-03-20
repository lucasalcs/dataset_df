import numpy as np
from typing import Tuple, Dict, Optional, Union, List
import pandas as pd
import argparse
from pathlib import Path
import sys
import os.path

def compute_det_curve(target_scores: np.ndarray, nontarget_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Detection Error Trade-off (DET) curve.
    
    Args:
        target_scores: Array of scores for target (bona fide) samples
        nontarget_scores: Array of scores for non-target (deepfake) samples
        
    Returns:
        Tuple containing (false rejection rates, false acceptance rates, thresholds)
    """
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))
    
    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]
    
    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)
    
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))
    
    return frr, far, thresholds

def compute_eer(target_scores: np.ndarray, nontarget_scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and the corresponding threshold.
    
    Args:
        target_scores: Array of scores for target (bona fide) samples
        nontarget_scores: Array of scores for non-target (deepfake) samples
        
    Returns:
        Tuple containing (EER, threshold)
    """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]

def compute_min_dcf(target_scores: np.ndarray, 
                   nontarget_scores: np.ndarray, 
                   p_target: float = 0.01,
                   c_miss: float = 1.0,
                   c_fa: float = 1.0) -> Tuple[float, float]:
    """
    Compute minimum Detection Cost Function (minDCF) and the corresponding threshold.
    
    Args:
        target_scores: Array of scores for target (bona fide) samples
        nontarget_scores: Array of scores for non-target (deepfake) samples
        p_target: Prior probability of target
        c_miss: Cost of miss (false rejection)
        c_fa: Cost of false alarm (false acceptance)
        
    Returns:
        Tuple containing (minDCF, threshold)
    """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    
    # Compute DCF for each threshold
    dcf = c_miss * p_target * frr + c_fa * (1 - p_target) * far
    min_dcf = np.min(dcf)
    min_threshold = thresholds[np.argmin(dcf)]
    
    return min_dcf, min_threshold

def compute_metrics(target_scores: np.ndarray, 
                   nontarget_scores: np.ndarray,
                   p_target: float = 0.01,
                   c_miss: float = 1.0,
                   c_fa: float = 1.0) -> Dict:
    """
    Compute all metrics for given target and non-target scores.
    
    Args:
        target_scores: Array of scores for target (bona fide) samples
        nontarget_scores: Array of scores for non-target (deepfake) samples
        p_target: Prior probability of target
        c_miss: Cost of miss (false rejection)
        c_fa: Cost of false alarm (false acceptance)
        
    Returns:
        Dictionary containing all computed metrics
    """
    eer, eer_threshold = compute_eer(target_scores, nontarget_scores)
    min_dcf, min_dcf_threshold = compute_min_dcf(target_scores, nontarget_scores, 
                                                p_target, c_miss, c_fa)
    
    return {
        'eer': eer,
        'eer_threshold': eer_threshold,
        'min_dcf': min_dcf,
        'min_dcf_threshold': min_dcf_threshold,
        'num_target': len(target_scores),
        'num_nontarget': len(nontarget_scores)
    }

def load_scores_csv(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load scores from a CSV file.
    Expected format: CSV with columns 'score' and 'label' (0 for deepfake, 1 for bona fide)
    
    Args:
        file_path: Path to the scores file
        
    Returns:
        Tuple containing (target_scores, nontarget_scores)
    """
    df = pd.read_csv(file_path)
    
    # Split scores based on labels
    target_scores = df[df['label'] == 1]['score'].values
    nontarget_scores = df[df['label'] == 0]['score'].values
    
    return target_scores, nontarget_scores

def load_asvspoof_scores(score_file: str, cm_key_file: str, phase: str = 'eval') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and prepare scores from ASVspoof 2021 DF files.
    
    Args:
        score_file: Path to the ASVspoof score file
        cm_key_file: Path to the ASVspoof 2021 CM protocol file
        phase: Dataset phase ('progress', 'eval', or 'hidden_track')
        
    Returns:
        Tuple containing (target_scores, nontarget_scores)
    """
    if not os.path.isfile(score_file):
        print(f"{score_file} doesn't exist")
        sys.exit(1)
        
    if not os.path.isfile(cm_key_file):
        print(f"{cm_key_file} doesn't exist")
        sys.exit(1)
        
    if phase not in ['progress', 'eval', 'hidden_track']:
        print("phase must be either progress, eval, or hidden_track")
        sys.exit(1)
    
    # Load protocol and score files
    cm_data = pd.read_csv(cm_key_file, sep=' ', header=None)
    
    try:
        submission_scores = pd.read_csv(score_file, sep=' ', header=None, skipinitialspace=True)
        
        if len(submission_scores.columns) > 2:
            print('CHECK: submission has more columns (%d) than expected (2). Check for leading/ending blank spaces.' % len(submission_scores.columns))
            sys.exit(1)
    except Exception as e:
        print(f"Error loading ASVspoof score file: {e}")
        sys.exit(1)
    
    if len(submission_scores) != len(cm_data):
        print('CHECK: submission has %d of %d expected trials.' % (len(submission_scores), len(cm_data)))
        sys.exit(1)
        
    # Merge scores with metadata
    cm_scores = submission_scores.merge(cm_data[cm_data[7] == phase], left_on=0, right_on=1, how='inner')
    
    # Separate bonafide and spoof scores
    bona_cm = cm_scores[cm_scores[5] == 'bonafide']['1_x'].values
    spoof_cm = cm_scores[cm_scores[5] == 'spoof']['1_x'].values
    
    return bona_cm, spoof_cm

def is_asvspoof_format(file_path: str) -> bool:
    """
    Check if the file appears to be in ASVspoof format.
    
    Args:
        file_path: Path to the score file
        
    Returns:
        True if it matches ASVspoof format, False otherwise
    """
    try:
        # Read first few lines
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()[:5]]
        
        # Check if each line has exactly two elements (ID and score)
        # and the ID matches ASVspoof ID format
        for line in lines:
            parts = line.split()
            if len(parts) != 2:
                return False
            
            # Check if first element looks like an ASVspoof ID
            if not (parts[0].startswith('DF_') or parts[0].startswith('LA_') or parts[0].startswith('PA_')):
                return False
                
            # Check if second element can be converted to float
            try:
                float(parts[1])
            except ValueError:
                return False
                
        return True
    except Exception:
        return False

def evaluate_scores(file_path: str, 
                   p_target: float = 0.01,
                   c_miss: float = 1.0,
                   c_fa: float = 1.0,
                   cm_key_file: Optional[str] = None,
                   phase: str = 'eval') -> Dict:
    """
    Evaluate scores and compute all metrics.
    This function detects the format and uses appropriate loading method.
    
    Args:
        file_path: Path to the scores file
        p_target: Prior probability of target
        c_miss: Cost of miss (false rejection)
        c_fa: Cost of false alarm (false acceptance)
        cm_key_file: Path to the ASVspoof CM protocol file (if ASVspoof format)
        phase: Dataset phase (if ASVspoof format)
        
    Returns:
        Dictionary containing all computed metrics
    """
    # Detect format and load data accordingly
    if is_asvspoof_format(file_path) and cm_key_file:
        print("Detected ASVspoof 2021 DF format. Using appropriate data loader.")
        target_scores, nontarget_scores = load_asvspoof_scores(file_path, cm_key_file, phase)
    else:
        target_scores, nontarget_scores = load_scores_csv(file_path)
    
    # Use the common metrics computation function
    return compute_metrics(target_scores, nontarget_scores, p_target, c_miss, c_fa)

def main():
    parser = argparse.ArgumentParser(description='Evaluate deepfake detection scores')
    parser.add_argument('scores_file', type=str, help='Path to the scores file')
    parser.add_argument('--p-target', type=float, default=0.01, help='Prior probability of target')
    parser.add_argument('--c-miss', type=float, default=1.0, help='Cost of miss')
    parser.add_argument('--c-fa', type=float, default=1.0, help='Cost of false alarm')
    parser.add_argument('--cm-key-file', type=str, help='Path to the ASVspoof CM protocol file (for ASVspoof format)')
    parser.add_argument('--phase', type=str, default='eval', choices=['progress', 'eval', 'hidden_track'], 
                        help='Dataset phase (for ASVspoof format)')
    
    args = parser.parse_args()
    
    results = evaluate_scores(
        args.scores_file, 
        args.p_target, 
        args.c_miss, 
        args.c_fa,
        args.cm_key_file,
        args.phase
    )
    
    print("\nEvaluation Results:")
    print(f"EER: {results['eer']:.4f} (threshold: {results['eer_threshold']:.4f})")
    print(f"minDCF: {results['min_dcf']:.4f} (threshold: {results['min_dcf_threshold']:.4f})")
    print(f"Number of bona fide samples: {results['num_target']}")
    print(f"Number of deepfake samples: {results['num_nontarget']}")

if __name__ == "__main__":
    main() 