import os
import glob
import numpy as np
import torch
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, Constellation
from sionna.phy.channel import AWGN
from data_utils import *
from universal_utils import *

# AMC configurations
AMC_CONFIGS = [
    {"m": 2, "k": 2048, "n": 6144},  # QPSK 1/3
    {"m": 2, "k": 3072, "n": 6144},  # QPSK 1/2
    {"m": 4, "k": 3072, "n": 6144},  # 16QAM 1/2
    {"m": 4, "k": 4096, "n": 6144},  # 16QAM 2/3
    {"m": 6, "k": 4096, "n": 6144},  # 64QAM 2/3
    {"m": 6, "k": 4608, "n": 6144},  # 64QAM 3/4
]


def find_thresholds(target_ber=1e-4, num_trials=100, device="cpu", save_dir="./logs/"):
    thresholds = []
    channel = AWGN()
    for cfg in AMC_CONFIGS:
        m, k, n = cfg["m"], cfg["k"], cfg["n"]
        encoder = LDPC5GEncoder(k=k, n=n)
        decoder = LDPC5GDecoder(encoder, hard_out=True)
        constellation = Constellation("qam", num_bits_per_symbol=m)
        mapper = Mapper(constellation=constellation)
        demapper = Demapper(
            demapping_method="app", constellation=constellation, output="llr"
        )
        for snr_db in np.arange(0, 14, 0.5):
            noise_var = snr_db_to_noise_var(snr_db, k, n, m)
            bers = []
            for _ in range(num_trials):
                bits = torch.randint(0, 2, (1, k), dtype=torch.float32, device=device)
                coded = encoder(bits)
                symbols = mapper(coded)
                rx = channel(symbols, noise_var)
                llr = demapper(rx, noise_var / 2)
                decoded = decoder(llr)
                bers.append(torch.mean(torch.abs(decoded - bits)).item())
            avg_ber = np.mean(bers)
            if avg_ber < target_ber:
                print(f"Config m={m} k={k} n={n}: threshold = {snr_db} dB")
                threshold = {**cfg, "threshold_snr_db": snr_db, "ber": avg_ber}
                thresholds.append(threshold)
                break
    thresholds.sort(key=lambda x: x["threshold_snr_db"])
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"thresholds.json"), "w") as fp:
        json.dump(thresholds, fp, indent=2)
    print(f"Thresholds saved to {os.path.join(save_dir, f'thresholds.json')}")
    return thresholds


def select_config(thresholds, snr_db):
    """
    Select the highest-throughput config whose threshold <= snr_db.
    thresholds: list of dicts loaded from JSON, sorted by threshold_snr_db ascending.
    """
    best = None
    for cfg in thresholds:
        if snr_db >= cfg["threshold_snr_db"]:
            best = cfg
    return best  # None if snr_db is below all thresholds


def transmit_bitstream(
    bitstream, k, encoder, decoder, mapper, demapper, channel, noise_var, device="cpu"
):
    true_len = len(bitstream)  # remember original length before padding
    # Pad bitstream to a multiple of k
    remainder = true_len % k
    if remainder != 0:
        pad_len = k - remainder
        bitstream = np.concatenate([bitstream, np.zeros(pad_len, dtype=np.int8)])
    num_blocks = len(bitstream) // k
    bits = torch.tensor(
        bitstream[: num_blocks * k], dtype=torch.float32, device=device
    ).reshape(num_blocks, k)
    coded = encoder(bits)
    symbols = mapper(coded)
    rx = channel(symbols, noise_var)
    llr = demapper(rx, noise_var / 2)
    decoded = decoder(llr)
    decoded = decoded.cpu().numpy().astype(np.int8).reshape(-1)
    # Strip padding to recover original length
    decoded = decoded[:true_len]
    return decoded, symbols


def ldpc_experiment(
    data_dirs,
    thresholds,
    config,
    snr_db_list,
    q_list,
    temp_dir="./temp/",
    log_dir="./logs/",
    device="cpu",
):
    channel = AWGN()
    # Step 1: Preprocess dataset
    img_dir = os.path.join(temp_dir, "images")
    if os.path.exists(img_dir):
        print(
            f"Temp dir {img_dir} already exists. Skipping preprocessing."
        )  # Avoid re-processing
        image_paths = sorted(glob(os.path.join(img_dir, "*.png")))
    else:
        image_paths = preprocess_dataset(data_dirs, config, temp_dir=img_dir)
    # Step 2: BPG encoding
    bpg_dir = os.path.join(temp_dir, "bpg")
    if os.path.exists(bpg_dir):
        print(f"Temp dir {bpg_dir} already exists. Skipping BPG encoding/decoding.")
    else:
        encode_bpg(image_paths, q_list, temp_dir=bpg_dir)
    for snr_db in snr_db_list:
        # Step 3: Channel transmission
        amc_config = select_config(thresholds, snr_db)
        if amc_config is None:
            print(f"No suitable AMC config for SNR={snr_db} dB, skipping.")
            continue
        m, k, n = amc_config["m"], amc_config["k"], amc_config["n"]
        constellation = Constellation("qam", num_bits_per_symbol=m)
        mapper = Mapper(constellation=constellation)
        demapper = Demapper(
            demapping_method="app", constellation=constellation, output="llr"
        )
        encoder = LDPC5GEncoder(k=k, n=n, dtype=torch.float32)
        decoder = LDPC5GDecoder(encoder, hard_out=True)
        noise_var = snr_db_to_noise_var(snr_db, k, n, m)
        for q in q_list:
            q_dir_bpg = os.path.join(bpg_dir, f"q{q}", "bpg")
            for img_path in image_paths:
                file_name = os.path.splitext(os.path.basename(img_path))[0]
                bpg_path = os.path.join(q_dir_bpg, f"{file_name}.bpg")
                file_name_postfix = f"_m{m}_k{k}_n{n}_snr{snr_db}"
                bitstream = file_to_bitstream(bpg_path)
                post_channel_bitstream, symbols = transmit_bitstream(
                    bitstream,
                    k,
                    encoder,
                    decoder,
                    mapper,
                    demapper,
                    channel,
                    noise_var,
                    device=device,
                )
                bitstream_to_file(
                    post_channel_bitstream,
                    bpg_path.replace(f"{file_name}", f"{file_name}{file_name_postfix}"),
                )
        # Step 4: BPG decoding and metric computation
        file_name_postfix = f"_m{m}_k{k}_n{n}_snr{snr_db}"
        bpg_results = decode_bpg(
            image_paths,
            q_list,
            temp_dir="./temp/bpg",
            file_name_postfix=file_name_postfix,
        )
        bpg_metrics = compute_metrics(
            bpg_results,
            device=device,
            log_dir=log_dir,
            file_name_postfix=file_name_postfix,
        )


# Main
if __name__ == "__main__":
    homedir = "/home/matthewwang16czap/"
    data_dirs = [os.path.join(homedir, "datasets/Kodak/")]
    temp_dir = "./temp/"
    log_dir = "./logs/"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Run this function to find the SNR thresholds for each AMC config at the target BER of 1e-4.
    # This will help determine AMC config for different SNRs in the main experiment.
    if not os.path.exists(os.path.join(log_dir, f"thresholds.json")):
        thresholds = find_thresholds(target_ber=1e-4, num_trials=100, device=device)
    else:
        with open(os.path.join(log_dir, f"thresholds.json"), "r") as fp:
            thresholds = json.load(fp)

    class DotDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    config = DotDict({"image_dims": (3, 256, 256), "max_test_samples": 100})
    q_list = list(range(1, 52))
    snr_db_list = list(range(1, 14))
    cbr_list = [x / 100.0 for x in range(1, 14, 1)]
    ldpc_experiment(
        data_dirs,
        thresholds,
        config,
        snr_db_list,
        q_list,
        temp_dir=temp_dir,
        log_dir=log_dir,
        device=device,
    )
