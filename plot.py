"""Every figure in the paper, written to plots/. Run from the repo root."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401  (registers the science styles)
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter, NullLocator

OUT = "plots"
SEL, REP = range(25), range(25, 50)
T_VALUES = (512, 1024, 2048, 4096, 8192)
T_LABELS = ["512", "1K", "2K", "4K", "8K"]
SCHEDULES = ("static", "dynamic", "work_stealing")
SCHED_LABEL = {
    "static": "Static",
    "dynamic": "Dynamic",
    "work_stealing": "Work stealing",
}

plt.style.use(["science", "grid", "bright", "no-latex"])
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "lines.markersize": 5,
        "savefig.dpi": 300,
    }
)
BLUE, RED, GREEN, PURPLE, GREY = "#4477AA", "#EE6677", "#228833", "#AA3377", "#BBBBBB"
REF = "#555555"

# The original profile legend and colors; keep them unchanged.
SEGMENTS = ["line7", "line8", "line10", "line11", "others"]
SEG_LABEL = {
    "line7": r"Line 7 (Load $\mathbf{K}$, $\mathbf{V}$)",
    "line8": r"Line 8 (Compute $\mathbf{S}$)",
    "line10": r"Line 10 (Online softmax)",
    "line11": r"Line 11 (Compute $\mathbf{O}$ in inner loop)",
    "others": r"Others",
    "idle": r"Idle",
}
SEG_COLOR = {
    "line7": "#4C72B0",
    "line8": "#55A868",
    "line10": "#C44E52",
    "line11": "#8172B3",
    "others": "#CCB974",
    "idle": "#D9D9D9",
}


def load(name):
    df = pd.read_csv(f"outputs/{name}/runtime.csv")
    df = df[df["T"].isin(T_VALUES)].copy()
    df["impl"] = name
    if "M_bytes" not in df:
        df["M_bytes"] = 0
    return df


def best(names, extra=()):
    """Per (T, d), the config with the lowest median runtime on SEL, reported as medians on REP."""
    df = pd.concat([load(n) for n in names])
    keys = ["T", "d", "impl", "M_bytes"]
    sel = df[df.testset.isin(SEL)].groupby(keys).runtime.median().reset_index()
    pick = sel.loc[sel.groupby(["T", "d"]).runtime.idxmin(), keys]
    rep = df[df.testset.isin(REP)].merge(pick)
    cols = ["runtime", *extra]
    return (
        rep.groupby(keys)[cols]
        .median()
        .reset_index()
        .set_index(["T", "d"])
        .sort_index()
    )


def fa2_mt(causal, threads=(8, 16)):
    if causal:
        return [f"fa2_causal_cpp_mt_{s}_t{n}" for s in SCHEDULES for n in threads]
    return [f"fa2_cpp_mt_t{n}" for n in threads]


def sdpa(causal, threads=(8, 16)):
    return [f"sdpa{'_causal' if causal else ''}_t{n}" for n in threads]


def naive(causal):
    return ["naive_causal_cpp" if causal else "naive_attn_cpp"]


def case_name(causal):
    return "masked" if causal else "self"


def t_axis(ax):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(T_VALUES))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(T_LABELS)


def log_y(ax, ticks):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def ref(ax, y):
    ax.axhline(y, color=REF, lw=0.9, zorder=1)


def ref_handle(label):
    return Line2D([], [], color=REF, lw=0.9, label=label)


def handles_of(ax):
    return ax.get_legend_handles_labels()[0]


def save(fig, name, tight=True):
    if tight:
        fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.02)
    else:
        # SciencePlots defaults to savefig.bbox="tight"; keep the exact figure size so
        # every fixed-geometry image has the same width and panel scale in the paper.
        with plt.rc_context({"savefig.bbox": "standard"}):
            fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", name)


COL_W = 3.35  # paper column width (in)
COL_RC = {
    "font.size": 7.5,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.5,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.5,
}


def row_major(handles, ncol):
    """Reorder so a column-first legend reads row by row."""
    rows = -(-len(handles) // ncol)
    return [
        handles[r * ncol + c]
        for c in range(ncol)
        for r in range(rows)
        if r * ncol + c < len(handles)
    ]


# Fixed panel geometry (in), so every panel renders at the same size in the paper.
PANEL_W, PANEL_H = 1.50, 1.30
MARGIN_L, MARGIN_R, PANEL_GAP = 0.27, 0.03, 0.05
MARGIN_B, TITLE_H = 0.40, 0.18


def fixed_panels(legend_rows=0):
    legend_h = 0.04 + 0.145 * legend_rows if legend_rows else 0.02
    fig_h = MARGIN_B + PANEL_H + TITLE_H + legend_h
    fig = plt.figure(figsize=(COL_W, fig_h))
    axes = []
    for i in range(2):
        left = MARGIN_L + i * (PANEL_W + PANEL_GAP)
        axes.append(
            fig.add_axes(
                (left / COL_W, MARGIN_B / fig_h, PANEL_W / COL_W, PANEL_H / fig_h),
                sharey=axes[0] if axes else None,
            )
        )
    axes[1].tick_params(labelleft=False)
    return fig, axes


def fixed_legend(fig, handles, ncol):
    """Legend centered on the drawn panels including their tick labels."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(r) for ax in fig.axes]
    center = (min(b.x0 for b in boxes) + max(b.x1 for b in boxes)) / 2 / fig.bbox.width
    fig.legend(
        handles=row_major(handles, ncol),
        loc="upper center",
        bbox_to_anchor=(center, 1.0),
        ncol=ncol,
        handlelength=1.8,
        columnspacing=1.2,
        handletextpad=0.5,
        borderaxespad=0.1,
    )


def compare_lines(ax, causal, d, scale):
    for impl, fn, color in (("FA-2", fa2_mt, BLUE), ("SDPA", sdpa, RED)):
        for nt, ls, mk in ((8, "-", "o"), (16, "--", "s")):
            y = scale(best(fn(causal, (nt,))).runtime)
            ax.plot(
                T_VALUES,
                y.xs(d, level="d"),
                color=color,
                ls=ls,
                marker=mk,
                mfc=color if nt == 8 else "white",
                label=f"{impl}, {nt} threads",
            )
    y = scale(best(naive(causal)).runtime)
    ax.plot(
        T_VALUES, y.xs(d, level="d"), color=GREY, marker="D", label="Naive, 1 thread"
    )


def runtime_vs_sdpa(causal):
    with plt.rc_context(COL_RC):
        fig, axes = fixed_panels(legend_rows=0 if causal else 3)
        for ax, d in zip(axes, (64, 128)):
            compare_lines(ax, causal, d, lambda r: r * 1e3)
            t_axis(ax)
            log_y(ax, [0.3, 1, 3, 10, 30, 100, 300])
            ax.set_xlabel("Sequence length $T$")
            ax.set_title(f"$d = {d}$")
        if not causal:
            fixed_legend(fig, handles_of(axes[0]), ncol=2)
        save(fig, f"fa2_vs_sdpa_runtime_{case_name(causal)}.png")


def speedup_over_sdpa(causal):
    base = best(sdpa(causal)).runtime
    with plt.rc_context(COL_RC):
        fig, axes = fixed_panels(legend_rows=0 if causal else 3)
        for ax, d in zip(axes, (64, 128)):
            compare_lines(ax, causal, d, lambda r: base / r)
            ref(ax, 1.0)
            t_axis(ax)
            log_y(ax, [0.05, 0.1, 0.2, 0.5, 1, 2])
            ax.set_ylim(0.04, 2.6)
            ax.set_xlabel("Sequence length $T$")
            ax.set_title(f"$d = {d}$")
        if not causal:
            fixed_legend(
                fig, handles_of(axes[0]) + [ref_handle("Parity with SDPA")], ncol=2
            )
        save(fig, f"speedup_{case_name(causal)}.png")


def schedules():
    bs = {
        nt: {s: best([f"fa2_causal_cpp_mt_{s}_t{nt}"]) for s in SCHEDULES}
        for nt in (8, 16)
    }
    style = {"dynamic": (GREEN, "o"), "work_stealing": (BLUE, "s")}
    with plt.rc_context(COL_RC):
        fig, axes = fixed_panels(legend_rows=2)
        for ax, d in zip(axes, (64, 128)):
            ax.plot(
                T_VALUES,
                np.ones(len(T_VALUES)),
                color=PURPLE,
                marker="^",
                label="Static",
            )
            for nt, ls, fill in ((8, "-", True), (16, "--", False)):
                b = bs[nt]
                for s in ("dynamic", "work_stealing"):
                    color, mk = style[s]
                    ax.plot(
                        T_VALUES,
                        (b["static"].runtime / b[s].runtime).xs(d, level="d"),
                        color=color,
                        ls=ls,
                        marker=mk,
                        mfc=color if fill else "white",
                        label=f"{SCHED_LABEL[s]}, {nt} thr.",
                    )
            t_axis(ax)
            ax.set_xlabel("Sequence length $T$")
            ax.set_title(f"$d = {d}$")
        h = handles_of(axes[0])
        # Rows: 8-thread lines and static, then 16-thread lines.
        fixed_legend(fig, [h[1], h[2], h[0], h[3], h[4]], ncol=3)
        save(fig, "schedules.png")


HEAT_M_KIB = (32, 256, 1024)


def metric_grid(name, metric):
    """Mean of `metric` per (M, T) over all test sets and both d."""
    df = pd.read_csv(f"outputs/{name}/runtime.csv")
    df = df[df["T"].isin(T_VALUES)]
    g = df.groupby(["M_bytes", "T"])[metric].mean().unstack("T")
    return g.loc[[m * 1024 for m in reversed(HEAT_M_KIB)], list(T_VALUES)].to_numpy()


def heat_axes(ax):
    ax.grid(False)
    ax.minorticks_off()
    ax.tick_params(which="both", length=0, top=False, right=False)
    ax.set_xticks(range(len(T_VALUES)), T_LABELS)
    ax.set_yticks(range(len(HEAT_M_KIB)), [str(m) for m in reversed(HEAT_M_KIB)])


def annotate(ax, grid, fmt, dark):
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            ax.text(
                j,
                i,
                format(v, fmt),
                ha="center",
                va="center",
                fontsize=5.5,
                color="white" if dark(v) else "black",
            )


HEAT_L, HEAT_GAP, HEAT_PH = 0.28, 0.06, 0.90  # left margin (M ticks only), gap, height
HEAT_CB_GAP, HEAT_CB_W, HEAT_CB_R = 0.07, 0.06, 0.40  # color bar gap, width, label room
HEAT_B, HEAT_T = 0.32, 0.16  # bottom (ticks + T label), title room


def heat_strip(grids, titles, name, cmap, norm, fmt, dark, cbar_label, cbar_ticks=None):
    n = len(grids)
    pw = (COL_W - HEAT_L - (n - 1) * HEAT_GAP - HEAT_CB_GAP - HEAT_CB_W - HEAT_CB_R) / n
    fig_h = HEAT_B + HEAT_PH + HEAT_T
    with plt.rc_context(COL_RC):
        fig = plt.figure(figsize=(COL_W, fig_h))
        axes = []
        for i, (grid, title) in enumerate(zip(grids, titles)):
            left = HEAT_L + i * (pw + HEAT_GAP)
            ax = fig.add_axes(
                (left / COL_W, HEAT_B / fig_h, pw / COL_W, HEAT_PH / fig_h),
                sharey=axes[0] if axes else None,
            )
            im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
            heat_axes(ax)
            annotate(ax, grid, fmt, dark)
            ax.set_title(title)
            ax.set_xlabel("$T$")
            if i:
                ax.tick_params(labelleft=False)
            axes.append(ax)
        cb_left = HEAT_L + n * pw + (n - 1) * HEAT_GAP + HEAT_CB_GAP
        cax = fig.add_axes(
            (cb_left / COL_W, HEAT_B / fig_h, HEAT_CB_W / COL_W, HEAT_PH / fig_h)
        )
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(cbar_label)
        if cbar_ticks:
            cbar.ax.yaxis.set_major_locator(FixedLocator(cbar_ticks))
            cbar.ax.yaxis.set_major_formatter(
                FixedFormatter([f"{t:g}" for t in cbar_ticks])
            )
        cbar.ax.yaxis.set_minor_locator(NullLocator())
        cbar.ax.tick_params(which="both", right=True, left=False, length=2)
        save(fig, name)


def load_imbalance_heatmap():
    grids = {
        (nt, s): metric_grid(f"fa2_causal_cpp_mt_{s}_t{nt}", "load_imbalance")
        for nt in (8, 16)
        for s in SCHEDULES
    }
    vmin = max(1.0, min(np.nanmin(g) for g in grids.values()))
    vmax = max(np.nanmax(g) for g in grids.values())
    span = np.log(vmax) - np.log(vmin)
    for nt in (8, 16):
        heat_strip(
            [grids[(nt, s)] for s in SCHEDULES],
            [SCHED_LABEL[s] for s in SCHEDULES],
            f"load_imbalance_t{nt}.png",
            "YlOrRd",
            LogNorm(vmin=vmin, vmax=vmax),
            ".1f",
            lambda v: (np.log(max(v, vmin)) - np.log(vmin)) / span > 0.65,
            "Max / mean",
            [1, 2, 4, 8, 16],
        )


def steal_success_heatmap():
    grids = {
        nt: metric_grid(f"fa2_causal_cpp_mt_work_stealing_t{nt}", "steal_success_rate")
        for nt in (8, 16)
    }
    vmax = max(np.nanmax(g) for g in grids.values())
    heat_strip(
        list(grids.values()),
        [f"{nt} threads" for nt in grids],
        "steal_success_heatmap.png",
        "YlGnBu",
        plt.Normalize(0, vmax),
        ".3f",
        lambda v: v > vmax * 0.55,
        "Success rate",
        [0, 0.1, 0.2, 0.3],
    )


def stacked(ax, frame, segments):
    y = np.arange(len(frame), dtype=float)
    left = np.zeros(len(frame))
    for seg in segments:
        w = frame[seg].to_numpy()
        ax.barh(
            y,
            w,
            left=left,
            height=0.65,
            color=SEG_COLOR[seg],
            label=SEG_LABEL[seg],
            edgecolor="white",
            linewidth=0.6,
        )
        for k in np.flatnonzero(w >= 8):
            ax.text(
                left[k] + w[k] / 2,
                y[k],
                f"{w[k]:.0f}%",
                ha="center",
                va="center",
                fontsize=6.5,
                color="black" if seg in ("others", "idle") else "white",
            )
        left += w
    ax.set_yticks(y, frame.index)
    ax.set_ylim(y.max() + 0.5, -0.5)
    ax.set_xlim(0, 100)
    ax.tick_params(axis="y", which="both", length=0)
    ax.grid(False, axis="y")
    ax.tick_params(axis="x", which="both", top=False)
    ax.xaxis.set_minor_locator(NullLocator())


PROF_PITCH, PROF_L, PROF_R = 0.27, 0.44, 0.10  # inches per bar row, label room, right
PROF_XLAB, PROF_GAP = 0.18, 0.06  # x tick labels only; the caption names the axis


def profile_figure(frames, segments, name, legend=True):
    pw = COL_W - PROF_L - PROF_R
    title_h = 0.19 if any(frames) else 0.03
    legend_h = 0.5 if legend else 0.0
    heights = [len(f) * PROF_PITCH for f in frames.values()]
    fig_h = (
        legend_h
        + PROF_XLAB
        + sum(heights)
        + len(frames) * title_h
        + PROF_GAP * (len(frames) - 1)
    )
    with plt.rc_context(COL_RC):
        fig = plt.figure(figsize=(COL_W, fig_h))
        axes, top = [], fig_h
        for (title, frame), h in zip(frames.items(), heights):
            top -= title_h + h
            ax = fig.add_axes(
                (PROF_L / COL_W, top / fig_h, pw / COL_W, h / fig_h),
                sharex=axes[0] if axes else None,
            )
            stacked(ax, frame, segments)
            if title:
                ax.set_title(title)
            axes.append(ax)
            top -= PROF_GAP
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)
        if legend:
            fig.legend(
                handles=row_major(handles_of(axes[0]), 2),
                loc="lower center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=2,
                handlelength=1.2,
                fontsize=6.8,
                columnspacing=0.8,
                handletextpad=0.4,
                borderaxespad=0.1,
            )
        save(fig, name, tight=False)


# Self-attention (no mask) always runs the dynamic schedule.
MT_KERNELS = ("self", "static", "dynamic", "work_stealing")


PROFILE_PARTS = SEGMENTS + ["idle"]


def profile_shares():
    """Percent of total thread time per step, indexed by (threads, kernel, d, T, M_bytes)."""
    rows = []
    for nt in (8, 16):
        for k in MT_KERNELS:
            name = (
                f"fa2_cpp_mt_profile_t{nt}"
                if k == "self"
                else f"fa2_causal_cpp_mt_profile_{k}_t{nt}"
            )
            df = pd.read_csv(f"outputs/{name}/runtime.csv")
            df = df[df.testset.isin(REP)]
            assert (df.is_correct == 1).all()
            g = df.groupby(["d", "T", "M_bytes"])[
                ["runtime"] + SEGMENTS + ["idle"]
            ].median()
            g["kernel"], g["threads"] = k, nt
            rows.append(g.reset_index())
    agg = pd.concat(rows).set_index(["threads", "kernel", "d", "T", "M_bytes"])[
        PROFILE_PARTS
    ]
    return agg.div(agg.sum(axis=1), axis=0) * 100


def mt_profile(T=8192, M=32768):
    # Shares averaged over 8 and 16 threads.
    at = (
        profile_shares()
        .xs((T, M), level=("T", "M_bytes"))
        .groupby(["kernel", "d"])
        .mean()
    )

    frame = at.loc["self"]
    frame.index = [f"$d = {d}$" for d in frame.index]
    profile_figure({"": frame}, PROFILE_PARTS, "mt_profile_self.png", legend=False)

    scheds = ["static", "dynamic", "work_stealing"]
    frames = {}
    for d in (64, 128):
        frame = at.xs(d, level="d").loc[scheds]
        frame.index = [SCHED_LABEL[k].replace(" ", "\n") for k in scheds]
        frames[f"$d = {d}$"] = frame
    profile_figure(frames, PROFILE_PARTS, "mt_profile_masked.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for causal in (False, True):
        runtime_vs_sdpa(causal)
    for causal in (False, True):
        speedup_over_sdpa(causal)
    schedules()
    load_imbalance_heatmap()
    steal_success_heatmap()
    mt_profile()
