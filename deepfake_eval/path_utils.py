from pathlib import Path

def repo_root() -> Path:
    # Assuming this file is at deepfake_eval/path_utils.py
    # Parent is deepfake_eval/, parent.parent is the repo root.
    return Path(__file__).resolve().parent.parent

def data_path(*parts: str) -> Path:
    """Return an absolute path to a file/dir inside the ./data/ directory."""
    return repo_root() / "data" / Path(*parts)

def config_path(*parts: str) -> Path:
    """Return an absolute path to a file/dir inside the ./configs/ directory."""
    return repo_root() / "configs" / Path(*parts)

def models_path(*parts: str) -> Path:
    """Return an absolute path to a file/dir inside the ./models/ directory."""
    return repo_root() / "models" / Path(*parts)

def output_path(*parts: str) -> Path:
    """Return an absolute path to a file/dir inside the ./output/ directory."""
    return repo_root() / "output" / Path(*parts)

# Add more helpers as needed, e.g., for notebooks/, scripts/ 