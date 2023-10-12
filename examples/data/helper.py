"""
A collection of utility functions to load the data.
"""

import numpy as np

def load_npy_data(npy_file):
    """Load .npy file"""
    data = np.load(npy_file, allow_pickle=True).item()
    return data
