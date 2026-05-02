import os
import subprocess
import numpy as np
import torch
from PIL import Image
import json
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure as SSIM,
    MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM,
)
import lpips
from utils import *


def encode_decode_bpg(image_paths, q_list, temp_dir="./temp/bpg"):
    """
    For each q, encodes all images with bpgenc and decodes with bpgdec.
    Returns dict: { q -> [ {orig_path, rec_path, bpp} ] }
    """
    results = {}
    for q in q_list:
        q_dir_bpg = os.path.join(temp_dir, f"q{q}", "bpg")
        q_dir_rec = os.path.join(temp_dir, f"q{q}", "rec")
        os.makedirs(q_dir_bpg, exist_ok=True)
        os.makedirs(q_dir_rec, exist_ok=True)
        q_results = []
        for img_path in image_paths:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            bpg_path = os.path.join(q_dir_bpg, f"{stem}.bpg")
            rec_path = os.path.join(q_dir_rec, f"{stem}.png")
            # Encode
            subprocess.run(
                ["bpgenc", "-q", str(q), img_path, "-o", bpg_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if not os.path.exists(bpg_path):
                print(f"[Warning] bpgenc failed for {img_path} at q={q}")
                continue
            # BPP
            orig = Image.open(img_path)
            H, W = orig.size[1], orig.size[0]
            bpp = os.path.getsize(bpg_path) * 8 / (H * W)
            # Decode
            subprocess.run(
                ["bpgdec", bpg_path, "-o", rec_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if not os.path.exists(rec_path):
                print(f"[Warning] bpgdec failed for {bpg_path}")
                continue
            q_results.append({"orig_path": img_path, "rec_path": rec_path, "bpp": bpp})
        results[q] = q_results
        with open(os.path.join(temp_dir, f"results.json"), "w") as fp:
            json.dump(results, fp, indent=2)
        print(f"q={q}: encoded/decoded {len(q_results)}/{len(image_paths)} images")
    return results


def compute_metrics(bpg_results, device="cuda", log_dir="./logs"):
    """
    For each q, computes mean CBR, PSNR, SSIM, MS-SSIM, LPIPS.
    Saves results to JSON.
    """
    os.makedirs(log_dir, exist_ok=True)
    ssim_fn = SSIM(data_range=1.0).to(device)
    msssim_fn = MS_SSIM(data_range=1.0).to(device)
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    all_results = []
    for q, items in bpg_results.items():
        psnr_list, ssim_list, msssim_list, lpips_list, bpp_list = [], [], [], [], []
        for item in items:
            orig_t = load_image_tensor(
                item["orig_path"], device
            )  # (1,C,H,W) float [0,1]
            rec_t = load_image_tensor(item["rec_path"], device)
            bpp = item["bpp"]
            psnr = compute_psnr(orig_t, rec_t).item()
            ssim = ssim_fn(rec_t, orig_t).item()
            msssim = msssim_fn(rec_t, orig_t).item()
            lp = (
                lpips_fn(rec_t * 2 - 1, orig_t * 2 - 1).mean().item()
            )  # lpips expects [-1,1]
            bpp_list.append(bpp)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            msssim_list.append(msssim)
            lpips_list.append(lp)
        all_results.append(
            {
                "q": q,
                "bpp": float(np.mean(bpp_list)),
                "psnr": float(np.mean(psnr_list)),
                "ssim": float(np.mean(ssim_list)),
                "msssim": float(np.mean(msssim_list)),
                "lpips": float(np.mean(lpips_list)),
            }
        )
        print(
            f"q={q} | BPP={all_results[-1]['bpp']:.4f} | PSNR={all_results[-1]['psnr']:.2f}"
        )
    out_path = os.path.join(log_dir, f"bpg_metrics.json")
    with open(out_path, "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"bpg metrics saved to {out_path}")
    return all_results


def compute_fix_snr_capacity(bpg_metrics, snr_db, cbr_list, log_dir="./logs"):
    results = []
    for cbr in cbr_list:
        max_bpp = get_max_bpp(snr_db, cbr)
        best_psnr = 0
        best_point = {}
        for p in bpg_metrics:
            bpp = p["bpp"]
            psnr = p["psnr"]
            if bpp <= max_bpp:
                if psnr > best_psnr:
                    best_psnr = psnr
                    best_point = p
        best_point["cbr"] = cbr
        best_point["snr"] = snr_db
        results.append(best_point)
    out_path = os.path.join(log_dir, f"bpg_capacity_snr_{snr_db}.json")
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"bpg capacity at snr={snr_db} saved to {out_path}")
    return results


def compute_fix_cbr_capacity(bpg_metrics, snr_db_list, cbr, log_dir="./logs"):
    results = []
    for snr_db in snr_db_list:
        max_bpp = get_max_bpp(snr_db, cbr)
        best_psnr = 0
        best_point = None
        for p in bpg_metrics:
            bpp = p["bpp"]
            psnr = p["psnr"]
            if bpp <= max_bpp:
                if psnr > best_psnr:
                    best_psnr = psnr
                    best_point = p
        best_point["cbr"] = cbr
        best_point["snr"] = snr_db
        results.append(best_point)
    out_path = os.path.join(log_dir, f"bpg_capacity_cbr_{cbr}.json")
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"bpg capacity at cbr={cbr} to {out_path}")
    return results


if __name__ == "__main__":
    homedir = "/home/matthewwang16czap/"
    data_dirs = [os.path.join(homedir, "datasets/Kodak/")]
    temp_dir = "./temp/"
    log_dir = "./logs/"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class DotDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    config = DotDict({"image_dims": (3, 256, 256), "max_test_samples": 100})
    q_list = list(range(1, 52))

    snr_db_list = list(range(1, 13))
    cbr_list = [x / 100.0 for x in range(1, 13, 1)]

    # Step 1
    img_dir = os.path.join(temp_dir, "images")
    if os.path.exists(img_dir):
        print(
            f"Temp dir {img_dir} already exists. Skipping preprocessing."
        )  # Avoid re-processing
        image_paths = sorted(glob(os.path.join(img_dir, "*.png")))
    else:
        image_paths = preprocess_dataset(data_dirs, config, temp_dir=img_dir)
    # Step 2
    bpg_dir = os.path.join(temp_dir, "bpg")
    if os.path.exists(bpg_dir):
        print(
            f"Temp dir {bpg_dir} already exists. Skipping BPG encoding/decoding."
        )  # Avoid re-encoding
        with open(os.path.join(bpg_dir, "results.json"), "r") as fp:
            bpg_results = json.load(fp)
    else:
        bpg_results = encode_decode_bpg(image_paths, q_list, temp_dir=bpg_dir)
    # Step 3
    metrics_dir = os.path.join(log_dir, "bpg_metrics.json")
    if os.path.exists(metrics_dir):
        with open(metrics_dir, "r") as fp:
            bpg_metrics = json.load(fp)
    else:
        bpg_metrics = compute_metrics(bpg_results, device=device, log_dir=log_dir)

    # get capacity at fixed SNR=7
    compute_fix_snr_capacity(bpg_metrics, 10, cbr_list, log_dir=log_dir)

    # get capacity at fixed cbr=0.0
    compute_fix_cbr_capacity(bpg_metrics, snr_db_list, 0.0625, log_dir=log_dir)
