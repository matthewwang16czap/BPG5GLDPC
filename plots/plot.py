import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def plot_psnr_by_cbr(results_path, snr_list=None, metric="psnr", save_path=None):
    with open(results_path) as f:
        results = json.load(f)

    # Group by SNR
    by_snr = defaultdict(list)
    for r in results:
        by_snr[r["snr"]].append(r)

    if snr_list is None:
        snr_list = sorted(by_snr.keys())

    markers = ["v", "o", "*", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(6, 5))

    for i, snr in enumerate(snr_list):
        data = sorted(by_snr[snr], key=lambda x: x["cbr"])
        cbrs = [d["cbr"] for d in data]
        psnrs = [d[metric] for d in data]
        marker = markers[i % len(markers)]
        ax.plot(
            cbrs,
            psnrs,
            linestyle="--",
            marker=marker,
            color="black",
            label=f"SNR={snr}dB",
            markersize=6,
        )

    ax.set_xlabel("channel bandwidth ratio")
    ax.set_ylabel(f"{metric.upper()} (dB)" if metric == "psnr" else metric.upper())
    ax.set_title("Kodak dataset, AWGN channel")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# Usage
plot_bpg_ldpc(
    "results.json",
    snr_list=[0, 4, 10],  # match the paper's SNR values
    metric="psnr",
    save_path="bpg_ldpc.png",
)
