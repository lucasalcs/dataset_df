import pickle
import os
from pathlib import Path
from typing import Optional, Set, Tuple

def load_pickle_keys(filepath: str) -> Optional[Set[str]]:
    """Loads keys (stems) from a dictionary stored in a pickle file."""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            print(f"Error: Pickle file '{filepath}' does not contain a dictionary.")
            return None

        keys_set = set()
        for key in data.keys():
            try:
                # Convert key to string, then extract stem using pathlib
                key_stem = Path(str(key)).stem
                keys_set.add(key_stem)
            except Exception as path_e:
                print(f"Warning: Could not process key '{key}' as a path stem: {path_e}. Using original key.")
                # Fallback: use the key as is (or string representation if not hashable)
                try:
                    keys_set.add(str(key))
                except TypeError:
                    print(f"Warning: Key '{key}' is not hashable and could not be added. Skipping.")
        return keys_set
    except pickle.UnpicklingError:
        print(f"Error: Could not unpickle file '{filepath}'. Invalid pickle format.")
        return None
    except FileNotFoundError:
        print(f"Error: Pickle file not found at '{filepath}'")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading pickle file '{filepath}': {e}")
        return None

def load_metadata_filenames(filepath: str, filename_col: int = 1, delimiter: Optional[str] = None) -> Optional[Set[str]]:
    """
    Loads filenames (stems) from a specified column in a metadata text file.
    Adjust filename_col and delimiter based on the metadata format.
    """
    filenames = set()
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'): # Skip empty lines and comments
                    continue
                parts = line.split(delimiter) # None splits by whitespace
                if len(parts) > filename_col:
                    filename = parts[filename_col]
                    # Extract stem to match pickle keys (assuming they are stems)
                    filename_stem = Path(filename).stem
                    filenames.add(filename_stem)
                else:
                    print(f"Warning: Could not extract filename from line {line_num} in '{filepath}'. Expected at least {filename_col + 1} columns.")
        return filenames
    except FileNotFoundError:
        print(f"Error: Metadata file not found at '{filepath}'")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading metadata file '{filepath}': {e}")
        return None

def check_key_match(pickle_file: str, metadata_file: str, filename_col: int = 1, delimiter: Optional[str] = None, verbose: bool = True) -> bool:
    """
    Checks if keys (stems) in a pickle file match filenames (stems) in a metadata text file.

    Args:
        pickle_file: Path to the input pickle file (expecting a dictionary).
        metadata_file: Path to the metadata text file.
        filename_col: 0-based index of the column containing filenames in the metadata file.
        delimiter: Delimiter used in the metadata file (default: whitespace).
        verbose: If True, print detailed messages and mismatches.

    Returns:
        True if all keys match, False otherwise (including errors during loading).
    """
    if verbose:
        print(f"Comparing pickle keys from: {pickle_file}")
        print(f"With metadata filenames from: {metadata_file} (column index: {filename_col})")
        print("-" * 30)

    pickle_keys = load_pickle_keys(pickle_file)
    metadata_filenames = load_metadata_filenames(metadata_file, filename_col, delimiter)

    if pickle_keys is None or metadata_filenames is None:
        if verbose:
            print("Comparison aborted due to errors loading files.")
        return False # Indicate failure

    if verbose:
        print(f"Found {len(pickle_keys)} unique keys (stems) in pickle file.")
        print(f"Found {len(metadata_filenames)} unique filenames (stems) in metadata file.")

    pickle_only = pickle_keys - metadata_filenames
    metadata_only = metadata_filenames - pickle_keys

    success = not pickle_only and not metadata_only

    if verbose:
        if success:
            print("\nSuccess! All keys match the filenames.")
        else:
            print("\nMismatch detected!")
            if pickle_only:
                print(f"\n{len(pickle_only)} Keys (stems) found in pickle file but NOT in metadata file:")
                count = 0
                for key in sorted(list(pickle_only)):
                    print(f"- {key}")
                    count += 1
                    if count >= 20:
                        print(f"... and {len(pickle_only) - count} more.")
                        break

            if metadata_only:
                print(f"\n{len(metadata_only)} Filenames (stems) found in metadata file but NOT in pickle file:")
                count = 0
                for fname in sorted(list(metadata_only)):
                    print(f"- {fname}")
                    count += 1
                    if count >= 20:
                        print(f"... and {len(metadata_only) - count} more.")
                        break
        print("-" * 30)

    return success 

def convert_asvspoof_scores_txt_to_pkl(
    txt_score_file_path: str,
    output_pkl_path: str,
    audio_dir_prefix: str, # e.g., "ASVspoof2021_DF_eval/flac"
    audio_ext: str = ".flac",
    score_file_cols: tuple[int, int] = (0, 1), # (utt_id_col_idx, score_col_idx)
    score_file_delimiter: str | None = None # None for whitespace
) -> None:
    """
    Converts an ASVspoof-style TXT score file (utt_id score) to a PKL dictionary
    mapping {relative_audio_path: score}, suitable for deepfake_eval loaders.

    Args:
        txt_score_file_path: Path to the input .txt score file.
        output_pkl_path: Path to save the output .pkl file.
        audio_dir_prefix: The prefix for audio file relative paths
                             (e.g., "ASVspoof2021_DF_eval/flac").
        audio_ext: Audio file extension (e.g., ".flac").
        score_file_cols: Tuple indicating (utt_id_column_index, score_column_index) in the txt file.
        score_file_delimiter: Delimiter used in the score txt file. None for any whitespace.
    """
    scores_dict = {}
    print(f"Reading TXT score file: {txt_score_file_path}")
    try:
        with open(txt_score_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'): # Skip empty lines and comments
                    continue
                parts = line.split(score_file_delimiter)
                if len(parts) > max(score_file_cols):
                    utt_id = parts[score_file_cols[0]]
                    score_str = parts[score_file_cols[1]]
                    try:
                        score = float(score_str)
                        # Construct the relative path key using os.path.join for safety
                        relative_path = os.path.join(audio_dir_prefix, f"{utt_id}{audio_ext}")
                        scores_dict[relative_path] = score
                    except ValueError:
                        print(f"Warning: Could not parse score '{score_str}' for utt_id '{utt_id}' on line {line_num}. Skipping.")
                else:
                    print(f"Warning: Line {line_num} ('{line}') has too few columns (expected > {max(score_file_cols)} based on indices {score_file_cols}). Skipping.")
        
        if not scores_dict:
            print(f"Warning: No scores were loaded from {txt_score_file_path}.")
            # Depending on desired behavior, could raise an error or just proceed to write an empty pickle.
            # For now, it will create an empty pickle if no scores are found.

        # Ensure output directory exists
        Path(output_pkl_path).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Found {len(scores_dict)} scores. Saving to PKL: {output_pkl_path}")
        with open(output_pkl_path, 'wb') as f_pkl:
            pickle.dump(scores_dict, f_pkl)
        print(f"Successfully converted and saved to {output_pkl_path}")

    except FileNotFoundError:
        print(f"Error: TXT score file not found at '{txt_score_file_path}'")
        raise # Re-raise the exception to be handled by the caller
    except Exception as e:
        print(f"An unexpected error occurred during conversion: {e}")
        raise # Re-raise 