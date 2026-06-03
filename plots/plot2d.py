import json
import os
import sys
from collections import defaultdict
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

cond_formatters = {
    "snr": lambda v: f"SNR={v}dB",
    "cbr": lambda v: f"CBR={v}",
}


def plot_2d(
    metrics_sources,
    cond_list,
    cond_name="snr",
    x_axis="cbr",
    y_axis="psnr",
    title="",
    save_path=None,
):
    """
    Args:
        metrics_sources: list of dicts

        Example:
        metrics_sources = [
            {
                "path": "ldpc_metrics.json",
                "label": "BPG + LDPC",
                "linestyle": "-",
                "marker": "o",
            },
            {
                "path": "bpg_capacity.json",
                "label": "BPG Capacity",
                "linestyle": "--",
                "marker": "s",
            },
        ]
    """

    # Backward compatibility for single file
    if isinstance(metrics_sources, str):
        metrics_sources = [
            {
                "path": metrics_sources,
                "label": None,
                "linestyle": "--",
                "marker": "o",
            }
        ]
    cond_labels = [cond_formatters[cond_name](v) for v in cond_list]
    # Colors distinguish conditions
    cond_colors = plt.cm.tab10(range(len(cond_list)))
    fig, ax = plt.subplots(figsize=(6, 5))
    for source in metrics_sources:
        metrics_path = source["path"]
        src_label = source.get("label", "")
        linestyle = source.get("linestyle", "-")
        marker = source.get("marker", "o")
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        metrics_by_cond = defaultdict(list)
        for m in metrics:
            metrics_by_cond[m[cond_name]].append(m)
        for i, cond_val in enumerate(sorted(cond_list)):
            if cond_val not in metrics_by_cond:
                continue
            data = sorted(
                metrics_by_cond[cond_val],
                key=lambda x: x[x_axis],
            )
            x_data = [d[x_axis] for d in data]
            y_data = [d[y_axis] for d in data]
            ax.plot(
                x_data,
                y_data,
                color=cond_colors[i],
                linestyle=linestyle,
                marker=marker,
                markersize=6,
                label="_nolegend_",
            )
    # Legend 1: Conditions (colors)
    cond_handles = [
        Line2D(
            [0],
            [0],
            color=cond_colors[i],
            linestyle="-",
            linewidth=2,
            label=cond_labels[i],
        )
        for i in range(len(cond_list))
    ]
    legend1 = ax.legend(
        handles=cond_handles,
        loc="upper left",
        fontsize=9,
        title="Conditions",
    )
    ax.add_artist(legend1)
    # Legend 2: Sources (markers + styles)
    source_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=source.get("linestyle", "-"),
            marker=source.get("marker", "o"),
            linewidth=2,
            markersize=6,
            label=source.get("label", ""),
        )
        for source in metrics_sources
    ]
    ax.legend(
        handles=source_handles,
        loc="lower right",
        fontsize=9,
        title="Methods",
    )
    # labels and grid
    ax.set_xlabel(axis_names.get(x_axis, x_axis))
    ax.set_ylabel(axis_names.get(y_axis, y_axis))
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    log_dir = "./logs/"
    save_dir = "./figs/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    # Define marker choices here
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
        "cond_name": "snr",
        "cond_list": [1, 4, 7, 10, 13],
        "x_axis": "cbr",
        "y_axis": "psnr",
        "channel": "AWGN",
        "dataset": "Kodak",
        "img_size": "256",
    }
    plot_2d(
        metrics_sources=[
            {
                "path": os.path.join(log_dir, "ldpc.json"),
                "label": "BPG + LDPC",
                "linestyle": "-",
                "marker": AVAILABLE_MARKERS[0],
            },
            {
                "path": os.path.join(log_dir, "capacity.json"),
                "label": "BPG Capacity",
                "linestyle": "--",
                "marker": AVAILABLE_MARKERS[1],
            },
        ],
        cond_list=fig_configs["cond_list"],
        cond_name=fig_configs["cond_name"],
        x_axis=fig_configs["x_axis"],
        y_axis=fig_configs["y_axis"],
        title=f"{fig_configs['img_size']}x{fig_configs['img_size']} {fig_configs['dataset']} Dataset, {fig_configs['channel']} Channel",
        save_path=os.path.join(
            save_dir,
            f"{fig_configs['dataset']}_{fig_configs['img_size']}_{fig_configs['channel']}_{fig_configs['x_axis']}_{fig_configs['y_axis']}_2d.png",
        ),
    )
