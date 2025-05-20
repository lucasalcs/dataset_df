import os
import pickle
import pandas as pd
from typing import Dict, Union, List
from tqdm import tqdm

from .base_loader import BaseLoader
from deepfake_eval.path_utils import data_path

class ASVspoof2021DFLoader(BaseLoader):
    """
    Loader for ASVspoof2021 DF evaluation scores.

    dataset_name = 'ASVspoof2021_DF'
    """
    dataset_name = 'ASVspoof2021_DF'
    # Default protocol/metadata file under data/metadata
    default_metadata = 'trial_metadata.txt'

    @staticmethod
    def _read_protocol(trial_file: str) -> pd.DataFrame:
        """Read and filter the ASVspoof2021 protocol file for a given phase."""
        prot = pd.read_csv(
            trial_file,
            sep=r'\s+',
            header=None,
            comment='#'
        )
        if prot.shape[1] <= 7:
            raise ValueError(
                f"Protocol has only {prot.shape[1]} cols, needs at least 8"
            )
        return prot[[1, 5, 7]].rename(
            columns={1: 'basename', 5: 'df_class', 7: 'phase'}
        )

    @classmethod
    def from_pkl_dict(
        cls,
        score_pkl: str,
        model_name: str,
        metadata_file: str | None = None,
        phase: str = 'eval',
        ext: str = '.flac'
    ) -> 'ASVspoof2021DFLoader':
        """
        Build loader from:
          - score_pkl: pickle mapping relative_path -> score
          - model_name: name of the system
          - metadata_file: protocol/metadata file (defaults to data/metadata/<default_metadata>)
          - phase: 'progress', 'eval', or 'hidden_track'
          - ext: file extension (default '.flac')
        """
        # 1) Determine protocol/metadata file
        if metadata_file is None:
            metadata_file = str(data_path('metadata', cls.default_metadata))
        # Load & filter protocol
        prot = cls._read_protocol(metadata_file)
        prot = prot[prot['phase'] == phase].reset_index(drop=True)

        # 2) Load scores
        with open(score_pkl, 'rb') as f:
            score_dict: Dict[str, float] = pickle.load(f)

        # 3) Map basename+ext -> full relative_path
        basename_map = {
            os.path.basename(k): k
            for k in score_dict.keys()
        }

        # 4) Check for missing
        expected = [f"{bn}{ext}" for bn in prot['basename']]
        missing = set(expected) - set(basename_map.keys())
        if missing:
            missing_list = sorted(list(missing))
            raise KeyError(
                f"Missing {len(missing_list)} files: "
                f"{missing_list[:10]}{'...' if len(missing_list)>10 else ''}"
            )

        # 5) Build records
        records: List[Dict[str, Union[str, float]]] = []
        for _, row in tqdm(prot.iterrows(), total=len(prot), desc="Building records 2021 DF"):
            key = f"{row['basename']}{ext}"
            full_rel = basename_map[key]
            records.append({
                'relative_path': full_rel,
                'subset':        phase,
                'df_class':      row['df_class'],
                'score':         score_dict[full_rel],
            })

        df = pd.DataFrame(records)
        # Attach source file attributes
        df.attrs['metadata_source_file'] = os.path.basename(metadata_file)
        df.attrs['score_source_file'] = os.path.basename(score_pkl)
        return cls(df, model_name) 