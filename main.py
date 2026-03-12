import os
import glob
import subprocess
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
import matplotlib.pyplot as plt
import json
import pandas as pd
from scipy.interpolate import PchipInterpolator

from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, Constellation
from sionna.phy.channel import AWGN

from torchmetrics.image import (
    StructuralSimilarityIndexMeasure as SSIM,
    MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM,
)
from utils import *


# AMC configurations
AMC_CONFIGS = [
    {"m": 2, "k": 2048, "n": 6144},  # QPSK 1/3
    {"m": 2, "k": 3072, "n": 6144},  # QPSK 1/2
    {"m": 4, "k": 3072, "n": 6144},  # 16QAM 1/2
    {"m": 4, "k": 4096, "n": 6144},  # 16QAM 2/3
    {"m": 6, "k": 4096, "n": 6144},  # 64QAM 2/3
    {"m": 6, "k": 4608, "n": 6144},  # 64QAM 3/4
]


# BPG encode once
def encode_bpg_once(image_paths, q_list, temp_dir):
    encoded = {}
    os.makedirs(temp_dir, exist_ok=True)

    for q in q_list:
        encoded[q] = []
        for img_path in image_paths:
            temp_bpg = os.path.join(temp_dir, f"{q}.bpg")
            subprocess.run(
                ["bpgenc", "-q", str(q), img_path, "-o", temp_bpg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with open(temp_bpg, "rb") as f:
                file_bytes = f.read()
            bitstream = np.unpackbits(np.frombuffer(file_bytes, dtype=np.uint8))
            encoded[q].append(bitstream)
    return encoded


# Transmission simulation
def transmit_bitstream(
    bitstream, k, encoder, decoder, mapper, demapper, channel, noise_var
):
    num_blocks = len(bitstream) // k
    if num_blocks == 0:
        return None, None
    bits = bitstream[: num_blocks * k].reshape(num_blocks, k)
    bits = tf.cast(bits, tf.float32)
    coded = encoder(bits)
    symbols = mapper(coded)
    rx = channel(symbols, noise_var)
    llr = demapper(rx, noise_var)
    decoded = decoder(llr)
    decoded = tf.cast(decoded, tf.uint8).numpy().reshape(-1)
    return decoded, symbols


# Decode BPG
def decode_bpg(bits, temp_dir):
    rec_bpg = os.path.join(temp_dir, "rec.bpg")
    rec_png = os.path.join(temp_dir, "rec.png")
    decoded_bytes = np.packbits(bits)
    with open(rec_bpg, "wb") as f:
        f.write(decoded_bytes)
    subprocess.run(
        ["bpgdec", rec_bpg, "-o", rec_png],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not os.path.exists(rec_png):
        return None
    return rec_png


# Main experiment
def run_experiment(dataset, q_list, snr_list, temp_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssim_metric = SSIM(data_range=1.0).to(device)
    msssim_metric = MS_SSIM(data_range=1.0).to(device)
    channel = AWGN()
    image_paths = sorted(glob.glob(os.path.join(dataset, "*")))
    encoded = encode_bpg_once(image_paths, q_list, temp_dir)
    results = []

    for cfg in AMC_CONFIGS:
        m = cfg["m"]
        k = cfg["k"]
        n = cfg["n"]

        constellation = Constellation("qam", num_bits_per_symbol=m)
        mapper = Mapper(constellation=constellation)
        demapper = Demapper(
            demapping_method="app", constellation=constellation, output="llr"
        )
        encoder = LDPC5GEncoder(k=k, n=n, dtype=tf.float32)
        decoder = LDPC5GDecoder(encoder, hard_out=True)

        for snr in snr_list:
            noise_var = snr_db_to_noise_var(snr, k, n, m)
            for q in q_list:
                psnr_list = []
                cbr_list = []
                for img_idx, img_path in enumerate(image_paths):
                    bitstream = encoded[q][img_idx]
                    decoded, symbols = transmit_bitstream(
                        bitstream,
                        k,
                        encoder,
                        decoder,
                        mapper,
                        demapper,
                        channel,
                        noise_var,
                    )

                    # original image process
                    orig = Image.open(img_path).convert("RGB")
                    orig = torch.from_numpy(np.array(orig)).float() / 255
                    orig = orig.permute(2, 0, 1).unsqueeze(0).to(device)

                    # cbr = (np.prod(symbols.shape) * m) / (np.prod(orig.shape) * 8)
                    cbr = (np.prod(bitstream.shape) / m) / (
                        orig.shape[-1] * orig.shape[-2]
                    )
                    if decoded is None:
                        psnr_list.append(0)
                        cbr_list.append(cbr)
                        continue

                    rec_path = decode_bpg(decoded, temp_dir)
                    if rec_path is None:
                        psnr_list.append(0)
                        cbr_list.append(cbr)
                        continue
                    rec = Image.open(rec_path).convert("RGB")
                    rec = torch.from_numpy(np.array(rec)).float() / 255
                    rec = rec.permute(2, 0, 1).unsqueeze(0).to(device)
                    if orig.shape != rec.shape:
                        psnr_list.append(0)
                        cbr_list.append(cbr)
                        continue

                    psnr = compute_psnr(orig, rec).item()
                    psnr_list.append(psnr)
                    cbr_list.append(cbr)

                results.append(
                    {
                        "snr": snr,
                        "q": q,
                        "psnr": np.mean(psnr_list),
                        "cbr": np.mean(cbr_list),
                        "config": cfg,
                    }
                )

    with open(os.path.join("./logs", f"ldpc.json"), "w") as fp:
        json.dump(results, fp)
    return results


def preprocess_experiment_data(data_list):
    """
    Cleans data by choosing the highest PSNR for duplicate (SNR, CBR) pairs
    and removing suboptimal points (Pareto front) for each SNR.
    """
    df = pd.DataFrame(data_list)

    # Round CBR first so that near-identical values are merged in groupby
    df["cbr"] = df["cbr"].round(3)

    # 1. Take highest PSNR for same SNR and CBR
    df = df.groupby(["snr", "cbr"]).psnr.max().reset_index()

    # 2. Filter for optimal points (Pareto front: PSNR must increase with CBR)
    processed_dfs = []
    for snr, group in df.groupby("snr"):
        group = group.sort_values("cbr")
        optimal_points = []
        max_psnr_so_far = -float("inf")
        for _, row in group.iterrows():
            if row["psnr"] > max_psnr_so_far:
                optimal_points.append(row)
                max_psnr_so_far = row["psnr"]
        processed_dfs.append(pd.DataFrame(optimal_points))

    return pd.concat(processed_dfs).reset_index(drop=True)


def plot_psnr_vs_cbr(df, snr_list):
    plt.figure(figsize=(10, 6))

    for snr in snr_list:
        subset = df[df["snr"] == snr].sort_values("cbr")
        if subset.empty:
            continue

        x, y = subset["cbr"].values, subset["psnr"].values

        if len(x) >= 3:
            # PchipInterpolator prevents "wavy" oscillations
            x_smooth = np.linspace(x.min(), x.max(), 500)
            interp = PchipInterpolator(x, y)
            y_smooth = interp(x_smooth)
            plt.plot(x_smooth, y_smooth, label=f"SNR {snr} dB", linewidth=2)
        else:
            # Fallback to straight line if too few points
            plt.plot(x, y, "o-", label=f"SNR {snr} dB")

        plt.scatter(x, y, s=40, alpha=0.5)

    plt.xlabel("CBR (3-decimal rounded)")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR vs CBR (Monotonic Smooth Curve)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_psnr_vs_snr(df, cbr_list, tolerance=0.01):
    """
    Plots a line chart of PSNR vs SNR for specific CBR settings.
    Since CBR is often discrete, it finds the closest available CBR within a tolerance.
    """
    plt.figure(figsize=(10, 6), dpi=100)
    available_cbrs = df["cbr"].unique()

    for target_cbr in cbr_list:
        # Find the unique CBR in the data closest to the user's request
        closest_cbr = available_cbrs[np.argmin(np.abs(available_cbrs - target_cbr))]

        if abs(closest_cbr - target_cbr) > tolerance:
            print(f"Skipping CBR {target_cbr}: No match found within tolerance.")
            continue

        subset = df[df["cbr"] == closest_cbr].sort_values("snr")
        plt.plot(subset["snr"], subset["psnr"], "o-", label=f"CBR ≈ {closest_cbr:.4f}")

    plt.xlabel("SNR (dB)")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR vs SNR Performance")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()


# Main
if __name__ == "__main__":
    dataset = "/home/matthewwang16czap/datasets/Kodak"
    # q_list = list(range(1, 52))
    # snr_list = list(range(1, 14))
    q_list = [51, 40, 30]
    snr_list = [1, 4, 7]

    temp_dir = "./temp"
    results = run_experiment(dataset, q_list, snr_list, temp_dir)

    # with open(os.path.join("./logs", f"ldpc.json"), "r") as fp:
    #     results = json.load(fp)

    # 1. Preprocess as usual (rounding to 3 decimals and Pareto front)
    results = preprocess_experiment_data(results)

    # 2. Define your specific CBR threshold (your exact code)
    unique_cbrs = results["cbr"].unique()
    selected_cbrs = sorted(unique_cbrs[unique_cbrs < 0.125])

    # 3. Filter the DataFrame to only include these CBRs
    filtered_results = results[results["cbr"].isin(selected_cbrs)]

    plot_psnr_vs_cbr(filtered_results, [1, 4, 7, 10, 13])

    plot_psnr_vs_snr(filtered_results, selected_cbrs)
