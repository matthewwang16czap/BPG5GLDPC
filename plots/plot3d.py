import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

axis_names = {
    "psnr": "PSNR (dB)",
    "msssim": "MS-SSIM (dB)",
    "lpips": "LPIPS Score",
    "snr": "SNR (dB)",
    "cbr": "Channel Bandwidth Ratio",
}


def plot_3d(
    metrics_sources,
    y_axis="psnr",
    title="",
    save_path=None,
):
    """
    metrics_sources = [
        {
            "path": "...",
            "label": "...",
            "marker": "o",
        },
        ...
    ]
    """

    if isinstance(metrics_sources, str):
        metrics_sources = [
            {
                "path": metrics_sources,
                "label": None,
                "marker": "o",
            }
        ]
    source_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    for src_idx, source in enumerate(metrics_sources):
        metrics_path = source["path"]
        src_label = source.get("label", "")
        marker = source.get("marker", "o")
        color = source.get(
            "color",
            source_colors[src_idx % len(source_colors)],
        )
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        snr_vals = sorted(set(m["snr"] for m in metrics))
        cbr_vals = sorted(set(m["cbr"] for m in metrics))
        lookup = {(m["snr"], m["cbr"]): m[y_axis] for m in metrics}
        SNR, CBR = np.meshgrid(
            snr_vals,
            cbr_vals,
        )
        Z = np.array(
            [[lookup.get((snr, cbr), np.nan) for snr in snr_vals] for cbr in cbr_vals]
        )
        # wireframe
        ax.plot_wireframe(
            SNR,
            CBR,
            Z,
            color=color,
            linewidth=1,
            alpha=0.7,
        )
        # scatter points
        ax.scatter(
            [m["snr"] for m in metrics],
            [m["cbr"] for m in metrics],
            [m[y_axis] for m in metrics],
            marker=marker,
            color=color,
            s=30,
            depthshade=True,
        )
    # Method legend
    method_handles = [
        Line2D(
            [0],
            [0],
            color=source.get(
                "color",
                source_colors[i % len(source_colors)],
            ),
            marker=source.get("marker", "o"),
            linestyle="-",
            linewidth=2,
            markersize=6,
            label=source.get("label", ""),
        )
        for i, source in enumerate(metrics_sources)
    ]
    ax.legend(
        handles=method_handles,
        loc="upper left",
        fontsize=9,
        title="Methods",
    )
    ax.set_xlabel(axis_names["snr"])
    ax.set_ylabel(axis_names["cbr"])
    ax.set_zlabel(axis_names[y_axis])
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )
    plt.show()


if __name__ == "__main__":
    log_dir = "./logs/"
    save_dir = "./figs/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    AVAILABLE_MARKERS = [
        "o",
        "s",
        "^",
        "D",
        "v",
        "*",
        "P",
        "X",
        "h",
        "H",
        "8",
        "p",
        "x",
        "+",
    ]
    fig_configs = {
        "cond_list": [1, 4, 7, 10, 13],
        "y_axis": "psnr",
        "channel": "AWGN",
        "dataset": "Kodak",
        "img_size": "256",
    }
    plot_3d(
        metrics_sources=[
            {
                "path": os.path.join(log_dir, "ldpc.json"),
                "label": "BPG + LDPC",
                "marker": AVAILABLE_MARKERS[0],
            },
            {
                "path": os.path.join(log_dir, "capacity.json"),
                "label": "BPG Capacity",
                "marker": AVAILABLE_MARKERS[1],
            },
        ],
        y_axis=fig_configs["y_axis"],
        title=f"{fig_configs['img_size']}x{fig_configs['img_size']} {fig_configs['dataset']} Dataset, {fig_configs['channel']} Channel",
        save_path=os.path.join(
            save_dir,
            f"{fig_configs['dataset']}_{fig_configs['img_size']}_{fig_configs['channel']}_{fig_configs['y_axis']}_3d.png",
        ),
    )
