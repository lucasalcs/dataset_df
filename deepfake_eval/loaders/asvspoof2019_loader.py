import os
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, Union, List
from tqdm import tqdm

from .base_loader import BaseLoader
from deepfake_eval.path_utils import data_path

class ASVspoof2019LALoader(BaseLoader):
    """
    Loader for ASVspoof 2019 LA evaluation.

    dataset_name = 'ASVspoof2019_LA'
    """
    dataset_name = 'ASVspoof2019_LA'
    # Default protocol filename under data/metadata
    default_protocol = 'ASVspoof2019.LA.cm.eval.trl.txt'

    @staticmethod
    def _read_protocol(trial_file: str) -> pd.DataFrame:
        """Read the ASVspoof2019 trial file."""
        return pd.read_csv(
            trial_file,
            sep=' ',
            header=None,
            names=['prefix', 'utt_id', 'subset_ignore', 'attack_type', 'df_class']
        )

    @classmethod
    def from_pkl_dict(
        cls,
        score_pkl: str,
        model_name: str,
        trial_file: str | None = None,
        ext: str = '.flac'
    ) -> 'ASVspoof2019LALoader':
        """
        Build loader from:
          - score_pkl : pickle mapping relative_path -> score
          - model_name: identifier for this system
          - trial_file: protocol file (defaults to data/metadata/<default_protocol>)
          - ext        : file extension (default '.flac')
        """
        # 1) Determine protocol file
        if trial_file is None:
            # Use the default protocol in data/metadata
            trial_file = str(data_path('metadata', cls.default_protocol))
        prot = cls._read_protocol(trial_file).reset_index(drop=True)

        # 2) Load score dictionary
        with open(score_pkl, 'rb') as f:
            score_dict: Dict[str, float] = pickle.load(f)

        # 3) Normalize extension
        ext = ext if ext.startswith('.') else f'.{ext}'

        # 4) Map basename -> full relative_path
        basename_map = {os.path.basename(k): k for k in score_dict.keys()}

        # 5) Check missing utterances
        expected = [f"{utt}{ext}" for utt in prot['utt_id']]
        missing = set(expected) - set(basename_map.keys())
        if missing:
            missing_list = sorted(list(missing))
            raise KeyError(
                f"Missing {len(missing_list)} utterances: "
                f"{missing_list[:10]}{'...' if len(missing_list)>10 else ''}"
            )

        # 6) Build records
        records: List[Dict[str, Union[str, float]]] = []
        for _, row in tqdm(prot.iterrows(), total=len(prot), desc="Building records 2019 LA"):
            utt = row['utt_id']
            basename = f"{utt}{ext}"
            full_key = basename_map[basename]
            first_folder = full_key.split(os.sep, 1)[0]
            subset = first_folder.split('_')[-1]
            records.append({
                'relative_path': full_key,
                'subset':        subset,
                'df_class':      row['df_class'],
                'score':         score_dict[full_key],
            })

        df = pd.DataFrame(records)
        return cls(df, model_name) 