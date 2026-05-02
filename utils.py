from glob import glob
import logging
import os
import datetime
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt

from datasets import get_dataset


def snr_db_to_noise_var(snr_db, k, n, m):
    R = k / n
    snr = 10 ** (snr_db / 10)
    EsN0 = snr * m * R
    noise_var = 1 / (2 * EsN0)
    return noise_var


def compute_psnr(x, y):
    mse = torch.mean((x - y) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 10 * torch.log10(1.0 / mse)


def get_max_bpp(snr_db, cbr):
    snr = 10 ** (snr_db / 10)
    max_bpp = np.log2(1 + snr) * cbr
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


def preprocess_dataset(data_dirs, config, temp_dir="./temp/images"):
    """
    Saves preprocessed images as PNG to temp_dir for bpgenc.
    Returns list of saved PNG paths.
    """
    os.makedirs(temp_dir, exist_ok=True)
    dataset = get_dataset(data_dirs=data_dirs, config=config)
    saved_paths = []
    for i, item in enumerate(dataset):
        img_np = (item.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        out_path = os.path.join(temp_dir, f"{i:05d}.png")
        Image.fromarray(img_np).save(out_path)
        saved_paths.append(out_path)
    print(f"Saved {len(saved_paths)} preprocessed images to {temp_dir}")
    return saved_paths


def load_image_tensor(path, device):
    img = Image.open(path).convert("RGB")
    t = torch.tensor(np.array(img)).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0).to(device)


def plot_lines(x, y, z, xlabel="x", ylabel="y", zlabel="z"):
    """
    x: 1D array of shape (N,)
    y: 1D array of shape (M,)
    z: 2D array of shape (M, N)
       each row corresponds to one y setting
    """
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    assert z.shape == (
        len(y),
        len(x),
    ), f"z shape must be ({len(y)}, {len(x)}) but got {z.shape}"

    plt.figure()

    for i, y_val in enumerate(y):
        plt.plot(x, z[i], label=f"{ylabel} = {y_val}")

    plt.xlabel(xlabel)
    plt.ylabel(zlabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
