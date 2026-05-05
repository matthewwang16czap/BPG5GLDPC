import json
import os
import numpy as np
import matplotlib.pyplot as plt

axis_names = {
    "psnr": "PSNR (dB)",
    "msssim": "MS-SSIM (dB)",
    "lpips": "LPIPS Score",
    "snr": "SNR (dB)",
    "cbr": "channel bandwidth ratio",
}


def plot_3d(
    metrics_sources,
    y_axis="psnr",
    title="",
    save_path=None,
):
    if isinstance(metrics_sources, str):
        metrics_sources = [(metrics_sources, None, "o")]
    else:
        normalized = []
        for t in metrics_sources:
            normalized.append((t[0], t[1], t[2] if len(t) == 3 else "o"))
        metrics_sources = normalized
    source_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    for src_idx, (metrics_path, src_label, marker) in enumerate(metrics_sources):
        color = source_colors[src_idx % len(source_colors)]
        with open(metrics_path) as f:
            metrics = json.load(f)
        # build a 2D grid over (snr, cbr) 
        snr_vals = sorted(set(m["snr"] for m in metrics))
        cbr_vals = sorted(set(m["cbr"] for m in metrics))
        lookup = {(m["snr"], m["cbr"]): m[y_axis] for m in metrics}
        SNR, CBR = np.meshgrid(snr_vals, cbr_vals)  # shape: (n_cbr, n_snr)
        Z = np.array([
            [lookup.get((snr, cbr), np.nan) for snr in snr_vals]
            for cbr in cbr_vals
        ])
        # wireframe across both snr and cbr 
        ax.plot_wireframe(
            SNR, CBR, Z,
            color=color,
            linewidth=1,
            alpha=0.7,
            label=src_label,
        )
        # scatter points on top
        ax.scatter(
            [m["snr"] for m in metrics],
            [m["cbr"] for m in metrics],
            [m[y_axis] for m in metrics],
            marker=marker,
            color=color,
            s=30,
            zorder=5,
            depthshade=True,
        )
    ax.set_xlabel(axis_names["snr"])
    ax.set_ylabel(axis_names["cbr"])
    ax.set_zlabel(axis_names[y_axis])
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    
if __name__ == "__main__":
    log_dir = "./logs/"
    save_dir = "./figs/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    plot_3d(
        metrics_sources=[
            (os.path.join(log_dir, "ldpc_metrics.json"), "BPG + LDPC", "o"),
            (os.path.join(log_dir, "bpg_capacity.json"), "BPG Capacity", "^"),
        ],
        y_axis="psnr",
        title="Kodak dataset, AWGN channel",
        save_path=os.path.join(save_dir, "kodak_psnr_3d.png"),
    )