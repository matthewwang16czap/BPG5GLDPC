import os
import torch
import json
from universal_utils import *
from data_utils import *


def capacity_experiment(bpg_metrics, snr_db_list, cbr_list, log_dir="./logs"):
    capacity_results = []
    for snr_db in snr_db_list:
        for cbr in cbr_list:
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
            if best_point is None:
                continue
            capacity_result = {"snr": snr_db, "cbr": cbr, **best_point}
            capacity_results.append(capacity_result)
    capacity_results.sort(key=lambda x: (x["snr"], x["cbr"]))
    out_path = os.path.join(log_dir, f"bpg_capacity.json")
    with open(out_path, "w") as fp:
        json.dump(capacity_results, fp, indent=2)
    print(f"Save bpg capacity to {out_path}")
    return capacity_results


if __name__ == "__main__":
    homedir = "/home/matthewwang16czap/"
    data_dirs = [os.path.join(homedir, "datasets/Kodak/")]
    temp_dir = "./temp/"
    log_dir = "./logs/"
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    # Start experiment
    capacity_experiment(bpg_metrics, snr_db_list, cbr_list, log_dir=log_dir)
