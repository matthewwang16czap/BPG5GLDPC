import os
import torch
import json
from utils import *
from datasets_utils import *


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

    snr_db_list = list(range(1, 14))
    cbr_list = [x / 100.0 for x in range(1, 14, 1)]

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
        encode_bpg(image_paths, q_list, temp_dir=bpg_dir)
        bpg_results = decode_bpg(image_paths, q_list, temp_dir=bpg_dir)
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
