import json
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict

axis_names = {
    "psnr": "PSNR (dB)",
    "msssim": "MS-SSIM (dB)",
    "lpips": "LPIPS Score",
    "snr": "SNR (dB)",
    "cbr": "channel bandwidth ratio",
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
        metrics_sources: a single path string, or a list of (path, label, linestyle) tuples
                         e.g. [("ldpc_metrics.json", "BPG + LDPC", "-"),
                               ("bpg_capacity.json", "BPG + Capacity", "--")]
        cond_list:       list of condition values to plot
        cond_name:       the condition dimension (e.g. "snr")
        x_axis:          metric key for the x-axis
        y_axis:          metric key for the y-axis
        save_path:       if given, saves the figure here
    """
    # Normalize input
    if isinstance(metrics_sources, str):
        metrics_sources = [(metrics_sources, None, "--")]
    else:
        normalized = []
        for t in metrics_sources:
            if len(t) == 3:
                normalized.append((t[0], t[1], t[2]))
            else:
                normalized.append((t[0], t[1], "--"))
        metrics_sources = normalized
    cond_labels = [cond_formatters[cond_name](v) for v in cond_list]
    markers = ["v", "o", "*", "s", "^", "D"]
    source_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(6, 5))
    for src_idx, (metrics_path, src_label, linestyle) in enumerate(metrics_sources):
        color = source_colors[src_idx % len(source_colors)]
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics_by_cond = defaultdict(list)
        for m in metrics:
            metrics_by_cond[m[cond_name]].append(m)
        for i, cond_val in enumerate(sorted(cond_list)):
            data = sorted(metrics_by_cond[cond_val], key=lambda x: x[x_axis])
            x_data = [d[x_axis] for d in data]
            y_data = [d[y_axis] for d in data]
            ax.plot(
                x_data,
                y_data,
                linestyle=linestyle,
                marker=markers[i % len(markers)],
                color=color,
                label="_nolegend_",  # suppress default legend
                markersize=6,
            )
    # Legend 1: markers conditions
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[i % len(markers)],
            color="black",  # neutral color for marker legend
            linestyle="--",
            markersize=6,
            label=cond_labels[i],
        )
        for i in range(len(cond_list))
    ]
    legend1 = ax.legend(
        handles=marker_handles,
        loc="upper left",
        fontsize=9,
    )
    ax.add_artist(legend1)  # keep it when adding the second legend
    # Legend 2: colors sources
    color_handles = [
        Line2D(
            [0],
            [0],
            color=source_colors[j % len(source_colors)],
            linestyle=metrics_sources[j][2],
            linewidth=2,
            label=metrics_sources[j][1],
        )
        for j in range(len(metrics_sources))
    ]
    ax.legend(
        handles=color_handles,
        loc="lower right",
        fontsize=9,
    )
    ax.set_xlabel(axis_names[x_axis])
    ax.set_ylabel(axis_names[y_axis])
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    log_dir = "./logs/"
    save_dir = "./figs/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    plot_2d(
        metrics_sources=[
            (os.path.join(log_dir, "ldpc_metrics.json"), "BPG + LDPC", "-"),
            (os.path.join(log_dir, "bpg_capacity.json"), "BPG Capacity", "--"),
        ],
        cond_list=[1, 4, 7, 10, 13],
        cond_name="snr",
        x_axis="cbr",
        y_axis="psnr",
        title="Kodak dataset, AWGN channel",
        save_path=os.path.join(save_dir, "kodak_psnr_cbr.png"),
    )
