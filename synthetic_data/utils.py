import os
import logging
import random

import numpy as np
import torch
from geomloss import SamplesLoss


def setup_logging(output_dir, log_file):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_file)

    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create handlers
    file_handler = logging.FileHandler(log_path)
    console_handler = logging.StreamHandler()

    # Create formatters and add them to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


def compute_geom_loss(x, y, x_weights=None, y_weights=None):
    sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    if x_weights is None and y_weights is None:
        return sinkhorn(x, y)

    if x_weights is None:
        x_weights = sinkhorn.generate_weights(x)
        y_weights = torch.tensor(y_weights, dtype=torch.float32)
    if y_weights is None:
        y_weights = sinkhorn.generate_weights(y)
        x_weights = torch.tensor(x_weights, dtype=torch.float32)

    x_weights = x_weights / x_weights.sum()
    y_weights = y_weights / y_weights.sum()
    return sinkhorn(x_weights, x, y_weights, y)
