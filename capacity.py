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

    # Save results to JSON
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(os.path.join("./logs", f"capacity_rate_{current_time}.json"), "w") as fp:
        json.dump(results, fp)
    return results


def curve_fix_snr(rd_points, snr_db, cbr_list):
    C = channel_capacity_awgn(snr_db)
    psnr_curve = []
    for cbr in cbr_list:
        max_rate = C * cbr
        best_psnr = 0
        for p in rd_points:
            bpp = p["rate"]
            psnr = p["psnr"]
            if bpp <= max_rate:
                best_psnr = max(best_psnr, psnr)
        psnr_curve.append(best_psnr)
    return psnr_curve


def curve_fix_cbr(rd_points, cbr, snr_list):
    psnr_curve = []
    for snr_db in snr_list:
        C = channel_capacity_awgn(snr_db)
        max_rate = C * cbr
        best_psnr = 0
        for p in rd_points:
            bpp = p["rate"]
            psnr = p["psnr"]
            if bpp <= max_rate:
                best_psnr = max(best_psnr, psnr)
        psnr_curve.append(best_psnr)
    return psnr_curve


# Main
if __name__ == "__main__":
    dataset = "/home/matthewwang16czap/datasets/Kodak"
    q_list = list(range(1, 52))
    snr_list = list(range(1, 40))
    cbr_list = [x / 100.0 for x in range(1, 26, 2)]

    # print("Computing BPG rate-distortion curve...")
    # rd_points = compute_bpg_rd_curve(dataset, q_list)

    with open(os.path.join("./logs", f"capacity_rate.json"), "r") as fp:
        rd_points = json.load(fp)

    # snr_db = 7
    # psnr_curve = curve_fix_snr(rd_points, snr_db=snr_db, cbr_list=cbr_list)
    # plt.plot(cbr_list, psnr_curve)
    # plt.xlabel("CBR")
    # plt.ylabel("PSNR")
    # plt.title("PSNR vs CBR (SNR fixed)")
    # plt.savefig(f"./plots/psnr_vs_cbr_fixed_snr{snr_db}.png")
    # # plt.show()

    cbr = 0.0625
    psnr_curve = curve_fix_cbr(rd_points, cbr=cbr, snr_list=snr_list)
    plt.plot(snr_list, psnr_curve)
    plt.xlabel("SNR (dB)")
    plt.ylabel("PSNR")
    plt.title("PSNR vs SNR (CBR fixed)")
    plt.savefig(f"./plots/psnr_vs_snr_fixed_cbr{cbr}.png")
    # plt.show()
