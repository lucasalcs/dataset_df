import os
import pickle
import pandas as pd
from typing import Dict, Union, List, Optional, Any
from tqdm import tqdm

from .base_loader import BaseLoader
from deepfake_eval.path_utils import data_path

class BRSpeechDFLoader(BaseLoader):
    """
    Loader for the BRSpeech-DF evaluation scores.

    dataset_name = 'brspeech_df'
    """
    dataset_name = 'brspeech_df'
    # Default metadata CSVs under data/metadata/brspeech_df/
    default_bonafide_metadata_filename = 'brspeech_df/brspeech_meta.csv'
    default_spoof_metadata_filename = 'brspeech_df/synthetic_meta.csv'

    @classmethod
    def from_pickles(
        cls,
        bonafide_pkl: str,
        spoof_pkl: str,
        model_name: str,
        bonafide_metadata_file: Optional[str] = None,
        spoof_metadata_file: Optional[str] = None,
        subset_filter: Optional[str] = None,
        df_class_filter: Optional[str] = None,
        bonafide_metadata_conditions: Optional[Dict[str, Any]] = None,
        spoof_metadata_conditions: Optional[Dict[str, Any]] = None
    ) -> 'BRSpeechDFLoader':
        """
        Build loader from separate bonafide/spoof pickles and their respective metadata CSVs.

        Args:
          bonafide_pkl: Pickle of {relative_path: score} for bona-fide files.
          spoof_pkl: Pickle of {relative_path: score} for spoof files.
          model_name: Label for the scoring model (e.g. 'AASIST').
          bonafide_metadata_file: Path to bonafide metadata CSV 
                                  (defaults to data/metadata/brspeech_df/brspeech_meta.csv).
          spoof_metadata_file: Path to spoof metadata CSV 
                               (defaults to data/metadata/brspeech_df/synthetic_meta.csv).
          subset_filter: Optional string to filter by 'subset' column (e.g., 'test', 'dev').
          df_class_filter: Optional string to filter by 'df_class' column (e.g., 'bonafide', 'spoof').
          bonafide_metadata_conditions: Optional dictionary of conditions for bonafide samples.
                                        (e.g., {'is_brspeech_sample': True, 'corpus': 'brspeech'}).
                                        Keys must be column names in the bonafide metadata.
          spoof_metadata_conditions: Optional dictionary of conditions for spoof samples.
                                     (e.g., {'tts': 'xtts'}). Keys must be column names in the spoof metadata.
        """
        # 1) Determine metadata files
        if bonafide_metadata_file is None:
            bonafide_metadata_file_path = str(data_path('metadata', cls.default_bonafide_metadata_filename))
        else:
            bonafide_metadata_file_path = bonafide_metadata_file
        
        if spoof_metadata_file is None:
            spoof_metadata_file_path = str(data_path('metadata', cls.default_spoof_metadata_filename))
        else:
            spoof_metadata_file_path = spoof_metadata_file

        try:
            df_bonafide_meta_full = pd.read_csv(bonafide_metadata_file_path, sep='|')
        except FileNotFoundError:
            raise FileNotFoundError(f"Bonafide metadata file not found: {bonafide_metadata_file_path}")
        
        try:
            df_spoof_meta_full = pd.read_csv(spoof_metadata_file_path, sep='|')
        except FileNotFoundError:
            raise FileNotFoundError(f"Spoof metadata file not found: {spoof_metadata_file_path}")

        # Validate essential columns for bonafide metadata
        required_bonafide_cols = ['relative_path', 'subset', 'df_class']
        missing_bonafide_cols = [c for c in required_bonafide_cols if c not in df_bonafide_meta_full.columns]
        if missing_bonafide_cols:
            raise ValueError(f"Bonafide metadata CSV ({bonafide_metadata_file_path}) missing core columns: {missing_bonafide_cols}")

        # Validate essential columns for spoof metadata (including 'tts')
        required_spoof_cols = ['relative_path', 'subset', 'df_class', 'tts']
        missing_spoof_cols = [c for c in required_spoof_cols if c not in df_spoof_meta_full.columns]
        if missing_spoof_cols:
            raise ValueError(f"Spoof metadata CSV ({spoof_metadata_file_path}) missing core columns: {missing_spoof_cols}")
            
        # Concatenate bonafide and spoof metadata. 'tts' column will have NaN for bonafide entries.
        df_meta_full = pd.concat([df_bonafide_meta_full, df_spoof_meta_full], ignore_index=True)
        
        # Determine all columns required for filtering, plus base requirements for the final DataFrame structure
        base_required_cols = ['relative_path', 'subset', 'df_class', 'tts'] # 'tts' will be NaN for bonafide
        filter_related_cols = set()
        if bonafide_metadata_conditions:
            filter_related_cols.update(bonafide_metadata_conditions.keys())
        if spoof_metadata_conditions:
            filter_related_cols.update(spoof_metadata_conditions.keys())
            
        all_required_cols = list(set(base_required_cols) | filter_related_cols)

        missing_cols = [c for c in all_required_cols if c not in df_meta_full.columns]
        # 'tts' is base_required but might not be used by spoof_metadata_conditions, handle its potential absence if not filtering by it
        if 'tts' in missing_cols and (not spoof_metadata_conditions or 'tts' not in spoof_metadata_conditions):
            # If tts is missing but not explicitly needed for filtering, we can proceed but it won't be in df.
            # The BaseLoader might complain if 'tts' is listed as optional but not present.
            # For now, we let it pass if not actively used in spoof_conditions.
            # However, 'tts' is in base_required_cols, so this logic path means it's truly missing from metadata.
            # BaseLoader's `optional` list only contains 'tts'. If it's not in df, it's fine by BaseLoader.
            # The issue is if our `records.append` tries to .get('tts', None) and it's not in df_meta.columns.
            # `row.get('tts', None)` handles this. So, the main concern is if a filter *needs* 'tts'.
             pass # Let it be caught if a filter strictly needs it and it's missing.
        elif any(c in missing_cols for c in filter_related_cols): # If a column needed for active filter is missing
            raise ValueError(f"Metadata CSV missing columns required for active filters: {[c for c in filter_related_cols if c in missing_cols]}")
        elif any(c in missing_cols for c in ['relative_path', 'subset', 'df_class']): # Core columns (tts already checked implicitly)
            raise ValueError(f"Combined metadata missing core columns: {[c for c in ['relative_path', 'subset', 'df_class'] if c in missing_cols]}")


        # Start with a copy of the full metadata
        df_meta = df_meta_full.copy()

        # Apply global filters
        if subset_filter:
            df_meta = df_meta[df_meta['subset'] == subset_filter].copy()
            if df_meta.empty:
                print(f"Warning: No metadata after subset_filter: '{subset_filter}'")
                return cls(pd.DataFrame(columns=base_required_cols + ['score'] + list(filter_related_cols)), model_name)

        if df_class_filter:
            df_meta = df_meta[df_meta['df_class'] == df_class_filter].copy()
            if df_meta.empty:
                print(f"Warning: No metadata after df_class_filter: '{df_class_filter}'")
                return cls(pd.DataFrame(columns=base_required_cols + ['score'] + list(filter_related_cols)), model_name)

        # Apply conditional filters
        keep_mask = pd.Series(True, index=df_meta.index)

        if bonafide_metadata_conditions:
            is_bonafide_series = (df_meta['df_class'] == 'bonafide')
            bonafide_filter_mask = pd.Series(True, index=df_meta.index[is_bonafide_series]) # Apply only to bonafide subset for condition building
            
            temp_df_bonafide = df_meta[is_bonafide_series]

            current_condition_mask = pd.Series(True, index=temp_df_bonafide.index)
            for col, expected_val in bonafide_metadata_conditions.items():
                if col not in temp_df_bonafide.columns: # Should have been caught by missing_cols check
                     raise ValueError(f"Bonafide condition column '{col}' not found in metadata.")
                if isinstance(expected_val, list):
                    current_condition_mask &= temp_df_bonafide[col].isin(expected_val)
                else:
                    current_condition_mask &= (temp_df_bonafide[col] == expected_val)
            
            # Update keep_mask: for bonafide rows, they must satisfy the conditions. Non-bonafide are untouched here.
            keep_mask.loc[is_bonafide_series] = current_condition_mask


        if spoof_metadata_conditions:
            is_spoof_series = (df_meta['df_class'] == 'spoof')
            # Similar logic for spoof conditions
            temp_df_spoof = df_meta[is_spoof_series]

            current_condition_mask = pd.Series(True, index=temp_df_spoof.index)
            for col, expected_val in spoof_metadata_conditions.items():
                if col not in temp_df_spoof.columns: # Should have been caught
                    raise ValueError(f"Spoof condition column '{col}' not found in metadata.")
                if isinstance(expected_val, list): # e.g. tts_filter_include = ['tts1', 'tts2']
                    current_condition_mask &= temp_df_spoof[col].isin(expected_val)
                else: # e.g. tts_filter_include = 'tts1'
                    current_condition_mask &= (temp_df_spoof[col] == expected_val)
            
            keep_mask.loc[is_spoof_series] &= current_condition_mask # Apply AND logic for spoof conditions

        df_meta = df_meta[keep_mask].copy() # apply combined keep_mask
        
        if df_meta.empty:
            print(f"Warning: No metadata records found after applying all filters.")
            return cls(pd.DataFrame(columns=base_required_cols + ['score'] + list(filter_related_cols)), model_name)

        # 2) Load score dictionaries
        with open(bonafide_pkl, 'rb') as f:
            bon_dict: Dict[str, float] = pickle.load(f)
        with open(spoof_pkl, 'rb') as f:
            spoof_dict: Dict[str, float] = pickle.load(f)

        # Helper to get score for a file
        def get_score(rel: str, label: str) -> Union[float, None]:
            d = bon_dict if label == 'bonafide' else spoof_dict
            if rel in d:
                return d[rel]
            root, ext = os.path.splitext(rel)
            if ext.lower() in ['.wav', '.flac']:
                alt_ext = '.flac' if ext.lower() == '.wav' else '.wav'
                alt = root + alt_ext
                if alt in d:
                    return d[alt]
            return None

        # Collect records only for entries with scores from the filtered metadata
        records: List[Dict[str, Union[str, float, None]]] = []
        for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Building BRSpeech-DF records"):
            rel = row['relative_path']
            label = row['df_class']
            score = get_score(rel, label)
            if score is None:
                continue
            
            record_data = {
                'relative_path': rel,
                'subset':        row['subset'],
                'df_class':      label,
                'score':         score,
                'tts':           row.get('tts', None) 
            }
            # Add other relevant metadata columns that might have been used for filtering or are just useful
            for col_name in filter_related_cols:
                if col_name in row: # Ensure the column exists in the row (it should, due to earlier checks)
                    record_data[col_name] = row.get(col_name, None)
            
            records.append(record_data)

        if not records:
            print(f"Warning: No records with scores found after applying all filters.")
            df = pd.DataFrame(columns=base_required_cols + ['score'] + list(filter_related_cols))
        else:
            df = pd.DataFrame(records)
        
        df.attrs['bonafide_metadata_source_file'] = os.path.basename(bonafide_metadata_file_path)
        df.attrs['spoof_metadata_source_file'] = os.path.basename(spoof_metadata_file_path)
        df.attrs['bonafide_source_file'] = os.path.basename(bonafide_pkl)
        df.attrs['spoof_source_file'] = os.path.basename(spoof_pkl)
        return cls(df, model_name) 