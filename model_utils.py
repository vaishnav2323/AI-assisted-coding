# compatibility wrapper for original file name
# the repository has `model_until.py` but `app.py` imports `model_utils`.
# This module re-exports the necessary symbols.
from model_until import load_artifacts, fe_single

__all__ = ["load_artifacts", "fe_single"]
