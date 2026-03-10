import os
import glob
import subprocess
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
import matplotlib.pyplot as plt
import json

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
                    if decoded is None:
                        continue
                    rec_path = decode_bpg(decoded, temp_dir)
                    if rec_path is None:
                        continue

                    orig = Image.open(img_path).convert("RGB")
                    rec = Image.open(rec_path).convert("RGB")
                    orig = torch.from_numpy(np.array(orig)).float() / 255
                    rec = torch.from_numpy(np.array(rec)).float() / 255
                    orig = orig.permute(2, 0, 1).unsqueeze(0).to(device)
                    rec = rec.permute(2, 0, 1).unsqueeze(0).to(device)
                    if orig.shape != rec.shape:
                        continue

                    psnr = compute_psnr(orig, rec).item()
                    cbr = (np.prod(symbols.shape) * m) / (np.prod(orig.shape) * 8)

                    psnr_list.append(psnr)
                    cbr_list.append(cbr)

                if len(psnr_list) == 0:
                    continue

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


# Envelope
def compute_envelope(points):
    points = sorted(points, key=lambda x: x[0])
    env_cbr = []
    env_psnr = []
    best = -1
    for cbr, psnr in points:
        if psnr > best:
            env_cbr.append(cbr)
            env_psnr.append(psnr)
            best = psnr
    return env_cbr, env_psnr


def extract_curves(results, snr_list):
    curves = {}
    for snr in snr_list:
        pts = [(r["cbr"], r["psnr"]) for r in results if r["snr"] == snr]
        curves[snr] = compute_envelope(pts)
    return curves


# Plot
def plot_ldpc(curves):
    markers = {0: "*", 4: "o", 10: "v"}
    for snr, (cbr, psnr) in curves.items():
        plt.step(
            cbr,
            psnr,
            where="post",
            linestyle="--",
            color="black",
            marker=markers.get(snr, "o"),
            label=f"SNR={snr}dB",
        )

    plt.xlabel("Channel Bandwidth Ratio")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    plt.legend()
    plt.show()


# Main
if __name__ == "__main__":
    dataset = "/home/matthewwang16czap/datasets/Kodak"
    q_list = [51, 38, 25, 13, 1]
    snr_list = [1, 4, 7, 10, 13]

    temp_dir = "./temp"
    results = run_experiment(dataset, q_list, snr_list, temp_dir)
    curves = extract_curves(results, snr_list)
    plot_ldpc(curves)
