"""
A collection of utility functions to load the data.
"""

import numpy as np

def load_npy_datasets(npy_files: str | list[str]):
    """Load .npy files"""
    #
    npy_files = [ npy_files ] if isinstance(npy_files, str) else npy_files
    #
    datasets = []
    for file in npy_files:
        datasets.append( np.load(file, allow_pickle=True).item() )
    #
    return datasets[0] if len(datasets) == 1 else datasets
