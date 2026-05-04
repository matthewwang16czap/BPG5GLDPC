from glob import glob
import logging
import os
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def snr_db_to_noise_var(snr_db, k, n, m):
    R = k / n
    snr = 10 ** (snr_db / 10)
    EsN0 = snr * m * R
    noise_var = 1 / EsN0
    return noise_var


def compute_psnr(x, y):
    mse = torch.mean((x - y) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 10 * torch.log10(1.0 / mse)


def get_max_bpp(snr_db, cbr):
    snr = 10 ** (snr_db / 10)
    max_bpp = 2 * np.log2(1 + snr) * cbr  # bits per complex use (2 real dims)
    return max_bpp


def bpp_to_cbr(bpp, snr_db):
    snr = 10 ** (snr_db / 10)
    C = 2 * np.log2(1 + snr)  # bits per complex use (2 real dims)
    cbr = bpp / C
    return cbr


def setup_logger(log_dir="./logs", current_time=None):
    os.makedirs(log_dir, exist_ok=True)
    if current_time is None:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{current_time}.log")
    logger = logging.getLogger("bpg_test")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - INFO] %(message)s")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
