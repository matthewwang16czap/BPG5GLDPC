import os
import glob
import subprocess
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
import logging
import datetime
import json

from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, Constellation
from sionna.phy.channel import AWGN

from torchmetrics.image import (
    StructuralSimilarityIndexMeasure as SSIM,
    MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM,
)

from utils import *


def bpg_5g_ldpc_test(
    k=4096,
    n=6144,
    m=4,  # default 16QAM
    q=51,  # [0,51], q higher quality lower
    snr_db=7,
    dataset="./Kodak",
    out_dir="./temp",
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ssim_metric = SSIM(data_range=1.0).to(device)
    msssim_metric = MS_SSIM(data_range=1.0).to(device)

    encoder = LDPC5GEncoder(k=k, n=n, dtype=tf.float32)
    decoder = LDPC5GDecoder(encoder, hard_out=True)

    constellation = Constellation("qam", num_bits_per_symbol=m)
    mapper = Mapper(constellation=constellation)
    demapper = Demapper(
        demapping_method="app", constellation=constellation, output="llr"
    )
    channel = AWGN()

    noise_var = snr_db_to_noise_var(snr_db, k, n, m)
    image_paths = sorted(glob.glob(os.path.join(dataset, "*")))

    psnr_list = []
    ssim_list = []
    msssim_list = []
    cbr_list = []

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for img_path in image_paths:
        # Remove temp files
        temp_bpg = os.path.join(out_dir, "temp.bpg")
        temp_rec_bpg = os.path.join(out_dir, "temp_rec.bpg")
        temp_rec_png = os.path.join(out_dir, "temp_rec.png")
        for f in [temp_bpg, temp_rec_bpg, temp_rec_png]:
            if os.path.exists(f):
                os.remove(f)

        # BPG Encode
        subprocess.run(
            ["bpgenc", "-q", str(q), img_path, "-o", temp_bpg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with open(temp_bpg, "rb") as f:
            file_bytes = f.read()

        # Bytes → Bits
        bitstream = np.unpackbits(np.frombuffer(file_bytes, dtype=np.uint8))
        total_bits = len(bitstream)
        num_blocks = total_bits // k
        if num_blocks == 0:
            continue
        bitstream = bitstream[: num_blocks * k]
        bitstream = bitstream.reshape(num_blocks, k)
        bits = tf.cast(bitstream, dtype=tf.float32)

        # LDPC Encode
        coded = encoder(bits)

        # QAM
        symbols = mapper(coded)

        # AWGN
        rx = channel(symbols, noise_var)

        # Demap + Decode
        llr = demapper(rx, noise_var)
        decoded_bits = decoder(llr)
        decoded_bits = tf.cast(decoded_bits, tf.uint8)
        decoded_bits = decoded_bits.numpy().reshape(-1)

        # Bits → Bytes → BPG Decode
        decoded_bytes = np.packbits(decoded_bits)
        with open(temp_rec_bpg, "wb") as f:
            f.write(decoded_bytes)
        subprocess.run(
            ["bpgdec", temp_rec_bpg, "-o", temp_rec_png],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if not os.path.exists(temp_rec_png):
            psnr_list.append(0)
            ssim_list.append(0)
            msssim_list.append(0)
            cbr_list.append(0)
            continue  # decoding failed (cliff effect)

        # Load Images for Metrics
        orig = Image.open(img_path).convert("RGB")
        orig = torch.from_numpy(np.array(orig)).float() / 255.0
        orig = orig.permute(2, 0, 1).unsqueeze(0).to(device)

        rec = Image.open(temp_rec_png).convert("RGB")
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
        cbr = (np.prod(symbols.shape) * m) / (np.prod(orig.shape) * 8).item()

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        msssim_list.append(msssim)
        cbr_list.append(cbr)

        # print(
        #     f"{os.path.basename(img_path)} | PSNR: {psnr:.2f} | SSIM: {ssim:.4f} | MS-SSIM: {msssim:.4f}"
        # )
        # errors = np.sum(bitstream != decoded_bits.reshape(num_blocks, k))
        # ber = errors / bitstream.size
        # print(f"Bit Error Rate (BER): {ber:.6f}")

    # Final Results
    # print("\n====== Final Results ======")
    # print(f"Average PSNR: {np.mean(psnr_list):.4f}")
    # print(f"Average SSIM: {np.mean(ssim_list):.4f}")
    # print(f"Average MS-SSIM: {np.mean(msssim_list):.4f}")

    return (
        np.mean(psnr_list),
        np.mean(ssim_list),
        np.mean(msssim_list),
        np.mean(cbr_list),
    )


def bpg_5g_ldpc_test_full(
    snr_list,
    q_list,
    k=4096,
    n=6144,
    m=4,
    dataset="./Kodak",
    out_dir="./temp",
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
    logger.info(f"k={k}, n={n}, m={m}")
    logger.info(f"SNR list: {snr_list}")
    logger.info(f"Q list: {q_list}")

    # 2D result containers, shape = (len(q_list), len(snr_list)
    psnr_matrix = []
    ssim_matrix = []
    msssim_matrix = []
    cbr_matrix = []

    # Loop SNR and Q
    for snr in snr_list:
        psnr_row = []
        ssim_row = []
        msssim_row = []
        cbr_row = []
        for q in q_list:
            psnr, ssim, msssim, cbr = bpg_5g_ldpc_test(
                k=k,
                n=n,
                m=m,
                q=q,
                snr_db=snr,
                dataset=dataset,
                out_dir=out_dir,
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
    bpg_5g_ldpc_test_full(
        snr_list=[1, 4, 7, 10, 13],
        q_list=[51, 38, 25, 13, 1],
        # snr_list=[13],
        # q_list=[1],
        dataset="/home/matthewwang16czap/datasets/Kodak",
    )
    # with open("./logs/bpg.json", "r") as fp:
    #     results = json.load(fp)
    # plot_lines(
    #     results["cbr"][0],
    #     results["snr"],
    #     results["msssim"],
    #     xlabel="CBR",
    #     ylabel="SNR",
    #     zlabel="MS-SSIM",
    # )
    # plot_lines(
    #     results["cbr"][0],
    #     results["snr"],
    #     results["psnr"],
    #     xlabel="CBR",
    #     ylabel="SNR",
    #     zlabel="PSNR",
    # )
