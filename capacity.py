import os
import glob
import subprocess
import numpy as np
import torch
from PIL import Image
import json
import datetime
import logging
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure as SSIM,
    MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM,
)

from utils import *


def bpg_capacity_test(
    snr_db,
    q,
    dataset="./Kodak",
    out_dir="./temp",
):

    os.makedirs(out_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(dataset, "*")))
    psnr_list = []
    ssim_list = []
    msssim_list = []
    cbr_list = []

    C = channel_capacity_awgn(snr_db)

    for img_path in image_paths:
        # Remove temp files
        temp_bpg = os.path.join(out_dir, "temp.bpg")
        temp_png = os.path.join(out_dir, "temp.png")
        for f in [temp_bpg, temp_bpg]:
            if os.path.exists(f):
                os.remove(f)

        # BPG Encode
        subprocess.run(
            ["bpgenc", "-q", str(q), img_path, "-o", temp_bpg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if not os.path.exists(temp_bpg):
            psnr_list.append(0)
            ssim_list.append(0)
            msssim_list.append(0)
            cbr_list.append(0)
            continue

        # Compute Bitrate
        file_size_bytes = os.path.getsize(temp_bpg)
        total_bits = file_size_bytes * 8

        orig = Image.open(img_path).convert("RGB")
        orig_np = np.array(orig)
        H, W, C_img = orig_np.shape

        # bits per pixel (source rate)
        bpp = total_bits / (H * W)

        # For capacity comparison:
        # assume 1 channel use per pixel (normalized bandwidth)
        # so CBR = bits per pixel
        cbr = bpp
        cbr_list.append(cbr)

        # Capacity Check
        if cbr <= C:
            # perfect transmission
            subprocess.run(
                ["bpgdec", temp_bpg, "-o", temp_png],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            if not os.path.exists(temp_png):
                psnr_list.append(0)
                ssim_list.append(0)
                msssim_list.append(0)
                continue

            # Load Images for Metrics
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ssim_metric = SSIM(data_range=1.0).to(device)
            msssim_metric = MS_SSIM(data_range=1.0).to(device)

            orig = Image.open(img_path).convert("RGB")
            orig = torch.from_numpy(np.array(orig)).float() / 255.0
            orig = orig.permute(2, 0, 1).unsqueeze(0).to(device)

            rec = Image.open(temp_png).convert("RGB")
            rec = torch.from_numpy(np.array(rec)).float() / 255.0
            rec = rec.permute(2, 0, 1).unsqueeze(0).to(device)

            # Mismatch Check: If H and W are swapped (Common at low SNR/Kodak)
            if orig.shape != rec.shape:
                # Check if transposing fixes it
                if orig.shape[2] == rec.shape[3] and orig.shape[3] == rec.shape[2]:
                    rec = rec.transpose(2, 3)
                else:
                    # If still not matching (e.g. wrong crop), resize rec to match orig
                    rec = torch.nn.functional.interpolate(
                        rec, size=orig.shape[2:], mode="bilinear", align_corners=False
                    )

            # Metrics
            psnr = compute_psnr(orig, rec).item()
            ssim = ssim_metric(orig, rec).item()
            msssim = msssim_metric(orig, rec).item()

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            msssim_list.append(msssim)

        else:
            # capacity exceeded → impossible transmission
            psnr_list.append(0)
            ssim_list.append(0)
            msssim_list.append(0)

    return (
        np.mean(psnr_list),
        np.mean(ssim_list),
        np.mean(msssim_list),
        np.mean(cbr_list),
    )


# -------------------------------------------------------
# Sweep SNR and Q
# -------------------------------------------------------


def bpg_capacity_test_full(
    snr_list,
    q_list,
    dataset="./Kodak",
):

    # Logger setup
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

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

    logger.info("========== Start Test ==========")
    logger.info(f"bpg capacity")
    logger.info(f"SNR list: {snr_list}")
    logger.info(f"Q list: {q_list}")

    # 2D result containers, shape = (len(q_list), len(snr_list)
    psnr_matrix = []
    ssim_matrix = []
    msssim_matrix = []
    cbr_matrix = []

    for snr in snr_list:
        psnr_row = []
        ssim_row = []
        msssim_row = []
        cbr_row = []

        for q in q_list:
            psnr, ssim, msssim, cbr = bpg_capacity_test(
                snr_db=snr,
                q=q,
                dataset=dataset,
            )

            psnr_row.append(round(psnr, 3))
            ssim_row.append(round(ssim, 4))
            msssim_row.append(round(msssim, 4))
            cbr_row.append(round(cbr, 4))
        psnr_matrix.append(psnr_row)
        ssim_matrix.append(ssim_row)
        msssim_matrix.append(msssim_row)
        cbr_matrix.append(cbr_row)

    # Log FULL 2D matrices
    logger.info("===== FINAL 2D RESULTS =====")
    logger.info(f"SNR (columns): {snr_list}")
    logger.info(f"Q (rows): {q_list}")
    logger.info(f"PSNR Matrix:\n{psnr_matrix}")
    logger.info(f"SSIM Matrix:\n{ssim_matrix}")
    logger.info(f"MS-SSIM Matrix:\n{msssim_matrix}")
    logger.info(f"CBR Matrix:\n{cbr_matrix}")
    logger.info("========== Finish Test ==========")

    results = {
        "snr": snr_list,
        "q": q_list,
        "psnr": psnr_matrix,
        "ssim": ssim_matrix,
        "msssim": msssim_matrix,
        "cbr": cbr_matrix,
    }

    with open(os.path.join(log_dir, f"{current_time}.json"), "w") as fp:
        json.dump(results, fp)

    return results


if __name__ == "__main__":

    bpg_capacity_test_full(
        snr_list=[1, 4, 7, 10, 13],
        q_list=[51, 38, 25, 13, 1],
        dataset="/home/matthewwang16czap/datasets/Kodak",
    )
