import torch
import numpy as np
import matplotlib.pyplot as plt


def snr_db_to_noise_var(snr_db, k, n, m):
    snr_linear = 10 ** (snr_db / 10.0)
    R = k / n
    return 1.0 / (R * m * snr_linear)


def compute_psnr(x, y):
    mse = torch.mean((x - y) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 10 * torch.log10(1.0 / mse)


def channel_capacity_awgn(snr_db):
    snr_linear = 10 ** (snr_db / 10.0)
    return np.log2(1 + snr_linear)  # bits per channel use


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
