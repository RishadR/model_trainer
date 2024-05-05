"""
Misc. functions useful during model training
"""
import random
import numpy as np
import torch

# CONSTANTS
DATA_LOADER_INPUT_INDEX, DATA_LOADER_LABEL_INDEX, DATA_LOADER_EXTRA_INDEX = 0, 1, 2

def set_seed(seed):
    """Manually sets the seed for python internals, numpy, torch and CUDA

    Args:
        seed: SEED
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
