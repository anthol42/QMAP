from .compute_global_identity import compute_global_identity
from .compute_binary_mask import compute_binary_mask
from .create_edgelist import create_edgelist


import pwiden_engine

def get_cache_dir() -> str:
    """
    Get the cache directory used by the pwiden engine.

    :return: The cache directory path as a string.
    """
    return pwiden_engine.get_cache_dir()
