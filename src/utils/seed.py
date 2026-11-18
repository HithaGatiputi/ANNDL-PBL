"""Seed every random number generator we use, so runs are reproducible."""
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
