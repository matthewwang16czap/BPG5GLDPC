import os
import glob
import subprocess
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from utils import *
import datetime
import json


# Compute BPG Rate–Distortion Points
def compute_bpg_rd_curve(dataset, q_list, temp_dir="./temp"):
    os.makedirs(temp_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(dataset, "*")))
    results = []

    for q in q_list:
        print(f"Processing q={q}...")
        psnr_list = []
        rate_list = []
        for img_path in image_paths:
            temp_bpg = os.path.join(temp_dir, "temp.bpg")
            temp_png = os.path.join(temp_dir, "temp.png")
            for f in [temp_bpg, temp_png]:
                if os.path.exists(f):
                    os.remove(f)

            # BPG encode
            subprocess.run(
                ["bpgenc", "-q", str(q), img_path, "-o", temp_bpg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Check if encoding was successful
            if not os.path.exists(temp_bpg):
                rate_list.append(0)
                psnr_list.append(0)
                continue

            # Compute rate
            file_size = os.path.getsize(temp_bpg) * 8
            orig = Image.open(img_path).convert("RGB")
            orig_np = np.array(orig)
            H, W, C = orig_np.shape
            bpp = file_size / (H * W)
            rate_list.append(bpp)

            # Decode
            subprocess.run(
                ["bpgdec", temp_bpg, "-o", temp_png],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            rec = Image.open(temp_png).convert("RGB")
            orig_t = torch.tensor(orig_np).float() / 255
            rec_t = torch.tensor(np.array(rec)).float() / 255
            orig_t = orig_t.permute(2, 0, 1).unsqueeze(0)
            rec_t = rec_t.permute(2, 0, 1).unsqueeze(0)
            psnr = compute_psnr(orig_t, rec_t).item()
            psnr_list.append(psnr)

        results.append({"q": q, "rate": np.mean(rate_list), "psnr": np.mean(psnr_list)})
    return results


# Compute Optimal PSNR under Capacity
def compute_capacity_bound(rd_points, snr_list):
    optimal_psnr = []
    optimal_cbr = []

    for snr in snr_list:
        print(f"Processing SNR={snr} dB...")
        C = channel_capacity_awgn(snr)
        best_psnr = 0
        best_cbr = None
        for p in rd_points:
            rate = p["rate"]
            psnr = p["psnr"]
            if rate <= C:
                if psnr > best_psnr:
                    best_psnr = psnr
                    best_cbr = rate / C
        optimal_psnr.append(best_psnr)
        optimal_cbr.append(best_cbr)

    # Save results to JSON
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    optimal_results = {
        "snr_list": snr_list,
        "optimal_psnr": optimal_psnr,
        "optimal_cbr": optimal_cbr,
    }
    with open(os.path.join("./logs", f"{current_time}.json"), "w") as fp:
        json.dump(optimal_results, fp)

    return optimal_psnr, optimal_cbr


# Main
if __name__ == "__main__":

    dataset = "/home/matthewwang16czap/datasets/Kodak"

    q_list = list(range(1, 52))
    snr_list = list(range(1, 50))

    print("Computing BPG rate-distortion curve...")

    rd_points = compute_bpg_rd_curve(dataset, q_list)

    print("Computing Shannon capacity bound...")

    optimal_psnr, optimal_cbr = compute_capacity_bound(rd_points, snr_list)

    # ---------------------------------------------------
    # Plot PSNR vs SNR
    # ---------------------------------------------------

    plt.figure()
    plt.plot(snr_list, optimal_psnr, linewidth=2)
    plt.xlabel("SNR (dB)")
    plt.ylabel("PSNR (dB)")
    plt.title("Optimal BPG + Shannon Capacity Bound")
    plt.grid(True)
    plt.show()

    # ---------------------------------------------------
    # Plot PSNR vs Channel Bandwidth Ratio
    # ---------------------------------------------------

    plt.figure()
    plt.plot(optimal_cbr, optimal_psnr, linewidth=2)
    plt.xlabel("Channel Bandwidth Ratio (CBR)")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR vs Channel Bandwidth Ratio")
    plt.grid(True)
    plt.show()
