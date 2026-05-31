import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

LINEWIDTH  = 1.5
GRID_ALPHA = 0.7
X_TICKS    = [32, 128, 256, 384, 512, 640, 768, 896, 1024]
X_LIMITS   = (32, 1024)

PROFILE_SEGMENTS = ["line7", "line8", "line10", "line11", "others"]
PROFILE_LABELS   = {
    "line7":  r"Line 7 (Load $\mathbf{K}$, $\mathbf{V}$)",
    "line8":  r"Line 8 (Compute $\mathbf{S}$)",
    "line10": r"Line 10 (Online softmax)",
    "line11": r"Line 11 (Compute $\mathbf{O}$ in inner loop)",
    "others": r"Others",
}
PROFILE_COLORS   = {"line7": "#4C72B0", "line8": "#55A868", "line10": "#C44E52",
                    "line11": "#8172B3", "others": "#CCB974"}

SCHEDULES = ('static', 'dynamic', 'work_stealing')


def runtime_csv(name):
    return os.path.join("outputs", name, "runtime.csv")


def causal_mt_name(sched, nt):
    return f"fa2_causal_cpp_mt_{sched}_t{nt}"


def get_smart_ticks(y_data, candidates, pad_top_if_close=False, top_close_ratio=0.9, max_ticks=6):
    if not y_data:
        return candidates
    y_min, y_max = min(min(y_data), 90), max(y_data)
    low_tick  = next((c for c in reversed(candidates) if c <= y_min), candidates[0])
    high_tick = next((c for c in candidates if c >= y_max), candidates[-1])
    if pad_top_if_close:
        try:
            idx = candidates.index(high_tick)
            if high_tick > 0 and (y_max / high_tick) >= top_close_ratio and idx < len(candidates) - 1:
                high_tick = candidates[idx + 1]
        except ValueError:
            pass
    ticks = [c for c in candidates if low_tick <= c <= high_tick]
    if len(ticks) <= max_ticks:
        return ticks
    idxs = np.linspace(0, len(ticks) - 1, max_ticks, dtype=int).tolist()
    for required_tick in (1, 10):
        if required_tick in ticks:
            req_idx = ticks.index(required_tick)
            if req_idx not in idxs:
                interior = [p for p, idx in enumerate(idxs) if idx not in (0, len(ticks) - 1)]
                if interior:
                    repl_pos = min(interior, key=lambda p: abs(idxs[p] - req_idx))
                    idxs[repl_pos] = req_idx
    idxs = sorted(set(idxs))
    return [ticks[i] for i in idxs]


def load_runtime_data(nt):
    """Mean runtime of FA-2 (ST and MT) as a percentage of naive attention."""
    df_naive = pd.read_csv(runtime_csv("naive_attn_cpp"))

    def load_and_agg(target_name, col_name):
        df     = pd.read_csv(runtime_csv(target_name))
        merged = pd.merge(df, df_naive, on=["testset", "T", "d"], suffixes=("", "_cpp"))
        merged[col_name] = (merged["runtime"] / merged["runtime_cpp"]) * 100
        return merged.groupby(["T", "d", "M_bytes"])[col_name].mean().reset_index()

    agg_fa2_cpp    = load_and_agg("fa2_cpp",           "pct_runtime_fa2_cpp")
    agg_fa2_cpp_mt = load_and_agg(f"fa2_cpp_mt_t{nt}", "pct_runtime_fa2_cpp_mt")
    return agg_fa2_cpp, agg_fa2_cpp_mt


def get_unique_values(dfs, col):
    vals = set()
    for df in dfs:
        if df is not None and not df.empty:
            vals.update(df[col].unique())
    return sorted(vals)


def filter_data(df, t_val, d_val):
    return df[(df["T"] == t_val) & (df["d"] == d_val)]


def setup_axis(ax, t_val, d_val, y_logscale=False):
    ax.set_title(f"T={int(t_val)}, d={int(d_val)}", fontsize=9)
    ax.set_xlim(*X_LIMITS)
    ax.set_xticks(X_TICKS)
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(visible=True, which='major', linestyle='--', alpha=GRID_ALPHA)
    if y_logscale:
        ax.set_yscale('log')
    ax.yaxis.set_minor_locator(FixedLocator([]))


def set_y_ticks(ax, y_data, is_speedup=False):
    if is_speedup:
        candidates, suffix = [0.05, 0.1, 0.5, 1, 2.5, 5, 10, 20, 50], 'x'
    else:
        candidates, suffix = [0.5, 1, 5, 25, 50, 100, 200, 400, 800], '%'
    ticks = get_smart_ticks(y_data, candidates, pad_top_if_close=is_speedup, top_close_ratio=0.88)
    ax.set_ylim(min(ticks), max(ticks))
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{t}{suffix}' for t in ticks])


def _legend_layout(causal=False):
    if causal:
        return 0.050, 0.025, 0.12
    return 0.035, 0.020, 0.08


def add_legend_rows(fig, rows, top_y, step):
    kwargs = dict(fontsize=7, loc='lower center', columnspacing=0.6,
                  handlelength=1.5, frameon=False)
    for i, (handles, labels) in enumerate(rows):
        if not handles:
            continue
        fig.legend(handles, labels,
                   bbox_to_anchor=(0.52, top_y - i * step),
                   ncol=len(handles), **kwargs)



def _draw_pie(ax, row_data, pct_label, fontsize=6, radius=1.0):
    values = [max(float(row_data.iloc[0][s]), 0.0) for s in PROFILE_SEGMENTS]
    ax.pie(values,
           colors=[PROFILE_COLORS[s] for s in PROFILE_SEGMENTS],
           startangle=90, autopct=pct_label, pctdistance=0.65,
           wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
           textprops={'fontsize': fontsize, 'fontweight': 'bold', 'color': 'white'},
           radius=radius)
    ax.axis('equal')

PAPER_PALETTE = [
    '#E41A1C',  # red
    '#377EB8',  # blue
    '#4DAF4A',  # green
    '#984EA3',  # purple
    '#FF7F00',  # orange
    '#A65628',  # brown
    '#F781BF',  # pink
    '#999999',  # gray
]

PAPER_COLORS = {
    # speedup.png (4 curves)
    'naive_cpp':           PAPER_PALETTE[7],  # gray baseline
    'fa2_cpp':             PAPER_PALETTE[1],  # blue
    ('fa2_cpp_mt', 8):     PAPER_PALETTE[2],  # green
    ('fa2_cpp_mt', 16):    PAPER_PALETTE[0],  # red
    # causal_speedup.png (8 curves)
    'naive_causal_cpp':    PAPER_PALETTE[7],  # gray baseline
    'fa2_causal_st':       PAPER_PALETTE[1],  # blue
    ('static', 8):         PAPER_PALETTE[2],
    ('static', 16):        PAPER_PALETTE[0],
    ('dynamic', 8):        PAPER_PALETTE[3],
    ('dynamic', 16):       PAPER_PALETTE[4],
    ('work_stealing', 8):  PAPER_PALETTE[5],
    ('work_stealing', 16): PAPER_PALETTE[6],
}

OUTPUT_DIR = os.path.join("outputs", "plots")

PAPER_LABELS = {
    'naive_cpp':        'Standard Attention',
    'fa2_cpp':          'FA-2 (ST)',
    'fa2_cpp_mt':       'FA-2 (MT, {nt} threads)',
    'naive_causal_cpp': 'Masked Attention',
    'fa2_causal_st':    'Masked FA-2 (ST)',
    'static':           'Masked FA-2 (static, {nt} threads)',
    'dynamic':          'Masked FA-2 (dynamic, {nt} threads)',
    'work_stealing':    'Masked FA-2 (work-stealing, {nt} threads)',
}


def _color(key, nt=None):
    if nt is not None and (key, nt) in PAPER_COLORS:
        return PAPER_COLORS[(key, nt)]
    return PAPER_COLORS.get(key)


def plot_speedup(num_threads_list=(8, 16), y_logscale=True):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    per_thread = []
    for nt in num_threads_list:
        try:
            agg_fa2_cpp, agg_fa2_cpp_mt = load_runtime_data(nt)
        except FileNotFoundError as e:
            print(f"Error: missing CSV for num_threads={nt}: {e}")
            continue
        per_thread.append({'nt': nt, 'agg_fa2_cpp': agg_fa2_cpp,
                           'agg_fa2_cpp_mt': agg_fa2_cpp_mt})

    if not per_thread:
        print("No runtime data available; skipping paper speedup plot.")
        return

    aggs = []
    for td in per_thread:
        aggs.extend([td['agg_fa2_cpp'], td['agg_fa2_cpp_mt']])
    T_values = get_unique_values(aggs, "T")
    d_values = get_unique_values(aggs, "d")
    nrows, ncols = len(T_values), len(d_values)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5, nrows * 1.6),
                             sharex=True, sharey=False)
    axes = np.atleast_2d(axes)

    h_naive = None
    h_st    = None
    mt_handles = {}

    for row, t_val in enumerate(T_values):
        for col, d_val in enumerate(d_values):
            ax = axes[row, col]
            y_data = [1]

            # Speedups are relative to naive attention, so it sits at 1x.
            line = ax.axhline(y=1, color=_color('naive_cpp'), linestyle='--',
                              linewidth=LINEWIDTH, label=PAPER_LABELS['naive_cpp'])
            if h_naive is None:
                h_naive = line

            # FA-2 (ST): plot once using the first thread bucket's agg.
            agg_st = per_thread[0]['agg_fa2_cpp']
            s = filter_data(agg_st, t_val, d_val).sort_values("M_bytes")
            if not s.empty:
                y = 100.0 / s['pct_runtime_fa2_cpp']
                line, = ax.plot(s["M_bytes"] / 1024, y,
                                linewidth=LINEWIDTH,
                                color=_color('fa2_cpp'),
                                label=PAPER_LABELS['fa2_cpp'])
                if h_st is None:
                    h_st = line
                y_data.extend(y)

            # FA-2 (MT) per thread count.
            for td in per_thread:
                nt = td['nt']
                s = filter_data(td['agg_fa2_cpp_mt'], t_val, d_val).sort_values("M_bytes")
                if s.empty:
                    continue
                y = 100.0 / s['pct_runtime_fa2_cpp_mt']
                line, = ax.plot(s["M_bytes"] / 1024, y,
                                linewidth=LINEWIDTH,
                                color=_color('fa2_cpp_mt', nt),
                                label=PAPER_LABELS['fa2_cpp_mt'].format(nt=nt))
                mt_handles.setdefault(nt, line)
                y_data.extend(y)

            setup_axis(ax, t_val, d_val, y_logscale)
            set_y_ticks(ax, y_data, is_speedup=True)

    # One legend row: MT (most threads first), then ST, then the baseline.
    ordered_threads = sorted(mt_handles.keys(), reverse=True)
    row_h, row_l = [], []
    for nt in ordered_threads:
        row_h.append(mt_handles[nt])
        row_l.append(mt_handles[nt].get_label())
    if h_st is not None:
        row_h.append(h_st); row_l.append(h_st.get_label())
    if h_naive is not None:
        row_h.append(h_naive); row_l.append(h_naive.get_label())

    rows = [(row_h, row_l)]
    top_y, step, bottom_pad = _legend_layout(causal=False)
    fig.text(0.52, bottom_pad - 0.01, "Cache Budget M (KiB)",
             ha='center', va='center', fontsize=9)
    fig.text(0.02, 0.5, "Speedup (x-times faster)",
             ha='center', va='center', rotation='vertical', fontsize=9)
    add_legend_rows(fig, rows, top_y, step)
    plt.tight_layout(rect=[0.04, bottom_pad, 1, 0.98])
    plt.savefig(os.path.join(OUTPUT_DIR, "speedup.png"),
                dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close()

    _write_summary(per_thread)


def _write_summary(per_thread, top_n=3):
    lines = []
    if per_thread:
        agg_st = per_thread[0]['agg_fa2_cpp']
        lines.append("========== FA-2 (C++, ST) ==========")
        lines.extend(_format_runtime_extremes(agg_st, "pct_runtime_fa2_cpp",
                                              top_n=top_n))
        lines.append("")

    for td in per_thread:
        nt = td['nt']
        lines.append(f"========== FA-2 (C++, MT, {nt} threads) ==========")
        lines.extend(_format_runtime_extremes(td['agg_fa2_cpp_mt'],
                                              "pct_runtime_fa2_cpp_mt",
                                              top_n=top_n))
        lines.append("")

    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def _causal_extremes_sections(num_threads_list=(8, 16), top_n=3):
    naive_path = runtime_csv("naive_causal_cpp")
    if not os.path.exists(naive_path):
        return []
    df_naive = pd.read_csv(naive_path)
    sections = []
    for nt in num_threads_list:
        sched_data = _load_causal_schedules(nt, df_naive)
        for sched in SCHEDULES:
            agg = sched_data.get(sched, pd.DataFrame())
            if agg.empty:
                continue
            block = [f"========== Masked FA-2 ({sched}, {nt} threads) =========="]
            block.extend(_format_speedup_extremes(agg, "speedup", top_n=top_n))
            sections.append("\n".join(block))
    return sections


def _format_runtime_extremes(agg_df, col, top_n=3):
    out = []
    if agg_df is None or agg_df.empty:
        out.append("  (no data)")
        out.append("-" * 30)
        return out

    def fmt(row):
        speedup = 100.0 / row[col] if row[col] > 0 else float('inf')
        return (f"  T={int(row['T'])}, d={int(row['d'])}, "
                f"M={int(row['M_bytes']/1024)}KB, "
                f"Runtime: {row[col]:.2f}%, Speedup: {speedup:.2f}x")

    top    = agg_df.nsmallest(top_n, col)
    bottom = agg_df.nlargest(top_n, col)
    out.append(f"Top {len(top)} Best Performance Boosts:")
    out.extend(fmt(r) for _, r in top.iterrows())
    out.append("")
    out.append(f"Top {len(bottom)} Least Performance Boosts:")
    out.extend(fmt(r) for _, r in bottom.iterrows())
    out.append("-" * 30)
    return out


def _format_speedup_extremes(agg_df, col, top_n=3):
    out = []
    if agg_df is None or agg_df.empty:
        out.append("  (no data)")
        out.append("-" * 30)
        return out

    def fmt(row):
        return (f"  T={int(row['T'])}, d={int(row['d'])}, "
                f"M={int(row['M_bytes']/1024)}KB, "
                f"Speedup: {row[col]:.2f}x")

    top    = agg_df.nlargest(top_n, col)
    bottom = agg_df.nsmallest(top_n, col)
    out.append(f"Top {len(top)} Best Performance Boosts:")
    out.extend(fmt(r) for _, r in top.iterrows())
    out.append("")
    out.append(f"Top {len(bottom)} Least Performance Boosts:")
    out.extend(fmt(r) for _, r in bottom.iterrows())
    out.append("-" * 30)
    return out


def plot_profile():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    path_cpp = runtime_csv("fa2_cpp_profile")
    if not os.path.exists(path_cpp):
        print(f"Warning: missing profile CSV {path_cpp}")
        return

    df = pd.read_csv(path_cpp)
    agg = df.groupby(["T", "d", "M_bytes"], as_index=False)[PROFILE_SEGMENTS].mean()
    if agg.empty:
        print("Profile data is empty after aggregation.")
        return
    agg = agg.sort_values(["T", "d", "M_bytes"])

    segment_handles = [mpatches.Patch(facecolor=PROFILE_COLORS[s], label=PROFILE_LABELS[s])
                       for s in PROFILE_SEGMENTS]
    legend_labels = [h.get_label() for h in segment_handles]
    t_values = sorted(agg["T"].unique())
    d_values = sorted(agg["d"].unique())
    max_cols = 3
    n_cols_total = len(t_values) * max_cols

    def pct_label(pct):
        return f"{pct:.0f}%" if pct >= 5 else ""

    # One row of pies per d, grouped by T.
    fig, axes = plt.subplots(len(d_values), n_cols_total,
                             figsize=(4.0, len(d_values) * 0.8))
    axes = np.atleast_2d(axes)
    fig.patch.set_facecolor('white')

    for d_idx, d_val in enumerate(d_values):
        for t_idx, t_val in enumerate(t_values):
            subset = agg[(agg["T"] == t_val) & (agg["d"] == d_val)]
            unique_m = sorted(subset["M_bytes"].unique())[:max_cols]
            col_offset = t_idx * max_cols

            for m_idx in range(max_cols):
                global_col = col_offset + m_idx
                ax = axes[d_idx, global_col]
                ax.set_facecolor('white')
                if m_idx < len(unique_m):
                    m_val = unique_m[m_idx]
                    row_data = subset[subset["M_bytes"] == m_val]
                    _draw_pie(ax, row_data, pct_label, fontsize=3, radius=1.3)
                    if d_idx == 0:
                        ax.set_title(f"M={int(m_val/1024)} KiB", fontsize=3, pad=1)
                else:
                    ax.axis('off')
                if global_col == 0:
                    ax.set_ylabel(f"d = {int(d_val)}", fontsize=3, labelpad=3)

    plt.subplots_adjust(left=0.04, right=0.99, top=0.82, bottom=0.14,
                        wspace=0.05, hspace=0.0)
    fig.canvas.draw()

    for t_idx, t_val in enumerate(t_values):
        col_offset = t_idx * max_cols
        bbox_left = axes[0, col_offset].get_position()
        bbox_right = axes[0, col_offset + max_cols - 1].get_position()
        center_x = (bbox_left.x0 + bbox_right.x1) / 2
        fig.text(center_x - 0.003, 0.91, f"T={int(t_val)}",
                 ha='center', va='center', fontsize=4)
        if t_idx < len(t_values) - 1:
            bbox_next = axes[0, col_offset + max_cols].get_position()
            sep_x = (bbox_right.x1 + bbox_next.x0) / 2
            fig.add_artist(plt.Line2D([sep_x, sep_x], [0.16, 0.82],
                                      transform=fig.transFigure,
                                      color='gray', linewidth=0.75))

    fig.legend(segment_handles, legend_labels, loc='lower center',
               bbox_to_anchor=(0.52, 0.01),
               ncol=len(segment_handles), fontsize=3, frameon=False,
               handlelength=1.8, columnspacing=1.2)

    plt.savefig(os.path.join(OUTPUT_DIR, "profile.png"),
                dpi=600, bbox_inches='tight', pad_inches=0.03)
    plt.close()


def _mean_causal_speedup(name, df_naive):
    """Per-(T, d, M) mean speedup over masked naive attention, or an empty frame."""
    path = runtime_csv(name)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    merged = pd.merge(df, df_naive, on=["testset", "T", "d"],
                      suffixes=("", "_naive"))
    merged["speedup"] = merged["runtime_naive"] / merged["runtime"]
    return (merged.groupby(["T", "d", "M_bytes"])["speedup"]
                  .mean().reset_index())


def _load_causal_schedules(nt, df_naive):
    return {sched: _mean_causal_speedup(causal_mt_name(sched, nt), df_naive)
            for sched in SCHEDULES}


def plot_causal_speedup(num_threads_list=(8, 16), y_logscale=True):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    naive_path = runtime_csv("naive_causal_cpp")
    if not os.path.exists(naive_path):
        print(f"Warning: missing {naive_path}, skipping paper causal plot.")
        return
    df_naive = pd.read_csv(naive_path)

    agg_st = _mean_causal_speedup("fa2_causal_cpp", df_naive)

    per_thread = []
    for nt in num_threads_list:
        sched_data = _load_causal_schedules(nt, df_naive)
        if any(not v.empty for v in sched_data.values()):
            per_thread.append((nt, sched_data))

    if not per_thread and agg_st.empty:
        print("No causal data available; skipping paper causal plot.")
        return

    all_aggs = [agg_st] if not agg_st.empty else []
    for _, sd in per_thread:
        all_aggs.extend([v for v in sd.values() if not v.empty])
    T_values = get_unique_values(all_aggs, "T")
    d_values = get_unique_values(all_aggs, "d")
    nrows, ncols = len(T_values), len(d_values)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5, nrows * 1.6),
                             sharex=True, sharey=False)
    axes = np.atleast_2d(axes)

    h_baseline = None
    h_st       = None
    sched_handles = {}

    for row, t_val in enumerate(T_values):
        for col, d_val in enumerate(d_values):
            ax = axes[row, col]
            y_data = [1]

            line = ax.axhline(y=1, color=_color('naive_causal_cpp'), linestyle='--',
                              linewidth=LINEWIDTH,
                              label=PAPER_LABELS['naive_causal_cpp'])
            if h_baseline is None:
                h_baseline = line

            if not agg_st.empty:
                s = filter_data(agg_st, t_val, d_val).sort_values("M_bytes")
                if not s.empty:
                    line, = ax.plot(s["M_bytes"] / 1024, s["speedup"],
                                    linewidth=LINEWIDTH,
                                    color=_color('fa2_causal_st'),
                                    label=PAPER_LABELS['fa2_causal_st'])
                    if h_st is None:
                        h_st = line
                    y_data.extend(s["speedup"].tolist())

            for nt, sched_data in per_thread:
                for sched in SCHEDULES:
                    agg = sched_data[sched]
                    if agg.empty:
                        continue
                    s = filter_data(agg, t_val, d_val).sort_values("M_bytes")
                    if s.empty:
                        continue
                    line, = ax.plot(s["M_bytes"] / 1024, s["speedup"],
                                    linewidth=LINEWIDTH,
                                    color=_color(sched, nt),
                                    label=PAPER_LABELS[sched].format(nt=nt))
                    sched_handles.setdefault((sched, nt), line)
                    y_data.extend(s["speedup"].tolist())

            setup_axis(ax, t_val, d_val, y_logscale)
            set_y_ticks(ax, y_data, is_speedup=True)

    # One legend row per thread count, then a row for ST and the baseline.
    ordered_threads = sorted([nt for nt, _ in per_thread], reverse=True)
    rows = []
    for nt in ordered_threads:
        row_h = [sched_handles[(s, nt)] for s in SCHEDULES
                 if (s, nt) in sched_handles]
        if row_h:
            rows.append((row_h, [h.get_label() for h in row_h]))
    base_h = [h for h in (h_st, h_baseline) if h is not None]
    if base_h:
        rows.append((base_h, [h.get_label() for h in base_h]))

    top_y, step, bottom_pad = _legend_layout(causal=True)
    step *= 0.75
    top_y += 0.020
    fig.text(0.52, bottom_pad - 0.01, "Cache Budget M (KiB)",
             ha='center', va='center', fontsize=9)
    fig.text(0.02, 0.5, "Speedup (x-times faster)",
             ha='center', va='center', rotation='vertical', fontsize=9)
    add_legend_rows(fig, rows, top_y, step)

    plt.tight_layout(rect=[0.04, bottom_pad, 1, 0.98])
    plt.savefig(os.path.join(OUTPUT_DIR, "causal_speedup.png"),
                dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def write_causal_summary(num_threads_list=(8, 16),
                         T_vals=(256, 512, 1024, 2048, 4096, 8192),
                         M_kib_vals=(32, 256, 1024)):
    """Write speedup extremes, steal success rates and load imbalance to one file.

    Metrics are averaged over test sets and both d values.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sections = []

    sections.extend(_causal_extremes_sections(num_threads_list=num_threads_list))

    for nt in num_threads_list:
        path = runtime_csv(causal_mt_name("work_stealing", nt))
        if not os.path.exists(path):
            print(f"Missing {path}, skipping steal success rate for threads={nt}.")
            continue
        df = pd.read_csv(path)
        if "steal_success_rate" not in df.columns:
            continue
        sections.append(_format_metric_table(
            df, metric="steal_success_rate",
            title=f"Steal success rate, work_stealing, threads={nt}",
            T_vals=T_vals, M_kib_vals=M_kib_vals))

    for nt in num_threads_list:
        for sched in SCHEDULES:
            path = runtime_csv(causal_mt_name(sched, nt))
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            if "load_imbalance" not in df.columns:
                continue
            sections.append(_format_metric_table(
                df, metric="load_imbalance",
                title=f"Load imbalance (max / mean time), {sched}, threads={nt}",
                T_vals=T_vals, M_kib_vals=M_kib_vals))

    if not sections:
        print("No causal data available; skipping causal_summary.txt.")
        return

    out_path = os.path.join(OUTPUT_DIR, "causal_summary.txt")
    with open(out_path, "w") as f:
        f.write("\n\n".join(sections) + "\n")


def _format_metric_table(df, metric, title, T_vals, M_kib_vals):
    """Table of mean ± std per (M, T), pooling test sets and d values."""
    lines = [title, "Avg ± SD across testsets and d values:"]
    header = f"  {'M (KiB)':>8} | " + " | ".join(f"{'T='+str(t):>16}" for t in T_vals)
    sep    = f"  {'-'*8}-+-" + "-+-".join("-" * 16 for _ in T_vals)
    lines.append(header)
    lines.append(sep)
    for kib in M_kib_vals:
        m = kib * 1024
        cells = []
        for t in T_vals:
            sub = df[(df["M_bytes"] == m) & (df["T"] == t)]
            if sub.empty:
                cells.append(f"{'n/a':>16}")
            else:
                mean = sub[metric].mean()
                std  = sub[metric].std(ddof=0)
                cells.append(f"{mean:7.3f} ± {std:6.3f}")
        lines.append(f"  {kib:>8} | " + " | ".join(cells))
    return "\n".join(lines)


def _speedup_mean_std(target_name, df_ref):
    """Per-(T, d, M_bytes) mean and std of the per-testset speedup over df_ref."""
    path = runtime_csv(target_name)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    merged = pd.merge(df, df_ref, on=["testset", "T", "d"],
                      suffixes=("", "_ref"))
    merged["speedup"] = merged["runtime_ref"] / merged["runtime"]
    grouped = merged.groupby(["T", "d", "M_bytes"])["speedup"]
    return pd.DataFrame({
        "mean": grouped.mean(),
        "std":  grouped.std(ddof=0),
    }).reset_index()


def _plot_band(ax, x, mean, std, color, label, y_floor=1e-3, y_logscale=True):
    lower = mean - std
    upper = mean + std
    if y_logscale:
        lower = np.maximum(lower, y_floor)
    line, = ax.plot(x, mean, color=color, linewidth=LINEWIDTH, label=label)
    ax.fill_between(x, lower, upper, color=color, alpha=0.18, linewidth=0)
    return line


def plot_speedup_std(num_threads_list=(8, 16), y_logscale=True):
    """Same layout as plot_speedup, with mean ± std as shaded bands."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_naive = pd.read_csv(runtime_csv("naive_attn_cpp"))
    st_agg = _speedup_mean_std("fa2_cpp", df_naive)

    per_thread = []
    for nt in num_threads_list:
        mt_agg = _speedup_mean_std(f"fa2_cpp_mt_t{nt}", df_naive)
        if not mt_agg.empty:
            per_thread.append({'nt': nt, 'agg': mt_agg})

    if st_agg.empty and not per_thread:
        print("No runtime data; skipping speedup_std.png.")
        return

    all_aggs = ([st_agg] if not st_agg.empty else []) + [td['agg'] for td in per_thread]
    T_values = get_unique_values(all_aggs, "T")
    d_values = get_unique_values(all_aggs, "d")
    nrows, ncols = len(T_values), len(d_values)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5, nrows * 1.6),
                             sharex=True, sharey=False)
    axes = np.atleast_2d(axes)

    h_naive = h_st = None
    mt_handles = {}

    for row, t_val in enumerate(T_values):
        for col, d_val in enumerate(d_values):
            ax = axes[row, col]
            y_data = [1]

            line = ax.axhline(y=1, color=_color('naive_cpp'), linestyle='--',
                              linewidth=LINEWIDTH, label=PAPER_LABELS['naive_cpp'])
            if h_naive is None:
                h_naive = line

            if not st_agg.empty:
                s = filter_data(st_agg, t_val, d_val).sort_values("M_bytes")
                if not s.empty:
                    line = _plot_band(ax, s["M_bytes"] / 1024,
                                      s["mean"].to_numpy(), s["std"].to_numpy(),
                                      color=_color('fa2_cpp'),
                                      label=PAPER_LABELS['fa2_cpp'],
                                      y_logscale=y_logscale)
                    if h_st is None:
                        h_st = line
                    y_data.extend((s["mean"] + s["std"]).tolist())
                    y_data.extend((s["mean"] - s["std"]).tolist())

            for td in per_thread:
                nt = td['nt']
                s = filter_data(td['agg'], t_val, d_val).sort_values("M_bytes")
                if s.empty:
                    continue
                line = _plot_band(ax, s["M_bytes"] / 1024,
                                  s["mean"].to_numpy(), s["std"].to_numpy(),
                                  color=_color('fa2_cpp_mt', nt),
                                  label=PAPER_LABELS['fa2_cpp_mt'].format(nt=nt),
                                  y_logscale=y_logscale)
                mt_handles.setdefault(nt, line)
                y_data.extend((s["mean"] + s["std"]).tolist())
                y_data.extend((s["mean"] - s["std"]).tolist())

            setup_axis(ax, t_val, d_val, y_logscale)
            set_y_ticks(ax, y_data, is_speedup=True)

    ordered_threads = sorted(mt_handles.keys(), reverse=True)
    row_h, row_l = [], []
    for nt in ordered_threads:
        row_h.append(mt_handles[nt]); row_l.append(mt_handles[nt].get_label())
    if h_st is not None:
        row_h.append(h_st); row_l.append(h_st.get_label())
    if h_naive is not None:
        row_h.append(h_naive); row_l.append(h_naive.get_label())

    rows = [(row_h, row_l)]
    top_y, step, bottom_pad = _legend_layout(causal=False)
    fig.text(0.52, bottom_pad - 0.01, "Cache Budget M (KiB)",
             ha='center', va='center', fontsize=9)
    fig.text(0.02, 0.5, "Speedup (x-times faster)",
             ha='center', va='center', rotation='vertical', fontsize=9)
    add_legend_rows(fig, rows, top_y, step)
    plt.tight_layout(rect=[0.04, bottom_pad, 1, 0.98])
    plt.savefig(os.path.join(OUTPUT_DIR, "speedup_std.png"),
                dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def plot_causal_speedup_std(num_threads_list=(8, 16), y_logscale=True):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    naive_path = runtime_csv("naive_causal_cpp")
    if not os.path.exists(naive_path):
        print(f"Missing {naive_path}; skipping causal_speedup_std.png.")
        return
    df_naive = pd.read_csv(naive_path)

    st_agg = _speedup_mean_std("fa2_causal_cpp", df_naive)

    per_thread = []
    for nt in num_threads_list:
        sched_aggs = {sched: _speedup_mean_std(causal_mt_name(sched, nt), df_naive)
                      for sched in SCHEDULES}
        if any(not v.empty for v in sched_aggs.values()):
            per_thread.append((nt, sched_aggs))

    if st_agg.empty and not per_thread:
        print("No causal data; skipping causal_speedup_std.png.")
        return

    all_aggs = ([st_agg] if not st_agg.empty else [])
    for _, sd in per_thread:
        all_aggs.extend(v for v in sd.values() if not v.empty)
    T_values = get_unique_values(all_aggs, "T")
    d_values = get_unique_values(all_aggs, "d")
    nrows, ncols = len(T_values), len(d_values)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5, nrows * 1.6),
                             sharex=True, sharey=False)
    axes = np.atleast_2d(axes)

    h_baseline = h_st = None
    sched_handles = {}

    for row, t_val in enumerate(T_values):
        for col, d_val in enumerate(d_values):
            ax = axes[row, col]
            y_data = [1]

            line = ax.axhline(y=1, color=_color('naive_causal_cpp'), linestyle='--',
                              linewidth=LINEWIDTH,
                              label=PAPER_LABELS['naive_causal_cpp'])
            if h_baseline is None:
                h_baseline = line

            if not st_agg.empty:
                s = filter_data(st_agg, t_val, d_val).sort_values("M_bytes")
                if not s.empty:
                    line = _plot_band(ax, s["M_bytes"] / 1024,
                                      s["mean"].to_numpy(), s["std"].to_numpy(),
                                      color=_color('fa2_causal_st'),
                                      label=PAPER_LABELS['fa2_causal_st'],
                                      y_logscale=y_logscale)
                    if h_st is None:
                        h_st = line
                    y_data.extend((s["mean"] + s["std"]).tolist())
                    y_data.extend((s["mean"] - s["std"]).tolist())

            for nt, sched_aggs in per_thread:
                for sched in SCHEDULES:
                    agg = sched_aggs[sched]
                    if agg.empty:
                        continue
                    s = filter_data(agg, t_val, d_val).sort_values("M_bytes")
                    if s.empty:
                        continue
                    line = _plot_band(ax, s["M_bytes"] / 1024,
                                      s["mean"].to_numpy(), s["std"].to_numpy(),
                                      color=_color(sched, nt),
                                      label=PAPER_LABELS[sched].format(nt=nt),
                                      y_logscale=y_logscale)
                    sched_handles.setdefault((sched, nt), line)
                    y_data.extend((s["mean"] + s["std"]).tolist())
                    y_data.extend((s["mean"] - s["std"]).tolist())

            setup_axis(ax, t_val, d_val, y_logscale)
            set_y_ticks(ax, y_data, is_speedup=True)

    ordered_threads = sorted([nt for nt, _ in per_thread], reverse=True)
    rows = []
    for nt in ordered_threads:
        row_h = [sched_handles[(s, nt)] for s in SCHEDULES
                 if (s, nt) in sched_handles]
        if row_h:
            rows.append((row_h, [h.get_label() for h in row_h]))
    base_h = [h for h in (h_st, h_baseline) if h is not None]
    if base_h:
        rows.append((base_h, [h.get_label() for h in base_h]))

    top_y, step, bottom_pad = _legend_layout(causal=True)
    step *= 0.75
    top_y += 0.020
    fig.text(0.52, bottom_pad - 0.01, "Cache Budget M (KiB)",
             ha='center', va='center', fontsize=9)
    fig.text(0.02, 0.5, "Speedup (x-times faster)",
             ha='center', va='center', rotation='vertical', fontsize=9)
    add_legend_rows(fig, rows, top_y, step)

    plt.tight_layout(rect=[0.04, bottom_pad, 1, 0.98])
    plt.savefig(os.path.join(OUTPUT_DIR, "causal_speedup_std.png"),
                dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def _build_metric_grid(df, metric, T_vals, M_kib_vals):
    grid = np.full((len(M_kib_vals), len(T_vals)), np.nan)
    for i, kib in enumerate(M_kib_vals):
        for j, t in enumerate(T_vals):
            sub = df[(df["M_bytes"] == kib * 1024) & (df["T"] == t)]
            if not sub.empty:
                grid[i, j] = sub[metric].mean()
    return grid


def plot_steal_success_heatmap(num_threads_list=(8, 16),
                               T_vals=(256, 512, 1024, 2048, 4096, 8192),
                               M_kib_vals=(32, 256, 1024)):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    panels = []
    for nt in num_threads_list:
        path = runtime_csv(causal_mt_name("work_stealing", nt))
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "steal_success_rate" not in df.columns:
            continue
        panels.append((nt, _build_metric_grid(df, "steal_success_rate",
                                              T_vals, M_kib_vals)))
    if not panels:
        return

    M_display = list(reversed(M_kib_vals))
    vmax = max(np.nanmax(g) for _, g in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 2.6),
                             sharey=True)
    axes = np.atleast_1d(axes)
    im = None
    for ax, (nt, grid) in zip(axes, panels):
        disp = grid[::-1]
        im = ax.imshow(disp, cmap="YlGnBu", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(f"threads={nt}", fontsize=8)
        ax.set_xticks(range(len(T_vals))); ax.set_xticklabels(T_vals, fontsize=7)
        ax.set_yticks(range(len(M_display))); ax.set_yticklabels(M_display, fontsize=7)
        ax.set_xlabel("T", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("M (KiB)", fontsize=8)
        for i in range(disp.shape[0]):
            for j in range(disp.shape[1]):
                v = disp[i, j]
                if np.isnan(v):
                    continue
                color = "white" if v > vmax * 0.55 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color=color, fontsize=7)
    cbar = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02,
                        ticks=[0.0, 0.1, 0.2, 0.3])
    cbar.set_label("success rate", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    plt.savefig(os.path.join(OUTPUT_DIR, "causal_steal_success_heatmap.png"),
                dpi=200, bbox_inches="tight")
    plt.close()


def plot_load_imbalance_heatmap(num_threads_list=(8, 16),
                                T_vals=(256, 512, 1024, 2048, 4096, 8192),
                                M_kib_vals=(32, 256, 1024)):
    sched_labels = {'static': 'static', 'dynamic': 'dynamic',
                    'work_stealing': 'work-stealing'}
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    grids = {}
    for nt in num_threads_list:
        for sched in SCHEDULES:
            path = runtime_csv(causal_mt_name(sched, nt))
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            if "load_imbalance" not in df.columns:
                continue
            grids[(nt, sched)] = _build_metric_grid(df, "load_imbalance",
                                                   T_vals, M_kib_vals)
    if not grids:
        return

    vmin = max(1.0, min(np.nanmin(g) for g in grids.values()))
    vmax = max(np.nanmax(g) for g in grids.values())
    norm = LogNorm(vmin=vmin, vmax=vmax)
    M_display = list(reversed(M_kib_vals))
    nrows = len(num_threads_list)
    fig, axes = plt.subplots(nrows, 3, figsize=(11, 2.3 * nrows),
                             sharex=True, sharey="row")
    axes = np.atleast_2d(axes)

    im = None
    log_span = np.log(vmax) - np.log(vmin) if vmax > vmin else 1.0
    for ri, nt in enumerate(num_threads_list):
        for ci, sched in enumerate(SCHEDULES):
            ax = axes[ri, ci]
            grid = grids.get((nt, sched))
            if grid is None:
                ax.axis("off")
                continue
            disp = grid[::-1]
            im = ax.imshow(disp, cmap="YlOrRd", norm=norm, aspect="auto")
            if ri == 0:
                ax.set_title(sched_labels[sched], fontsize=8)
            if ci == 0:
                ax.set_ylabel(f"threads={nt}\nM (KiB)", fontsize=8)
            ax.set_xticks(range(len(T_vals))); ax.set_xticklabels(T_vals, fontsize=7)
            ax.set_yticks(range(len(M_display))); ax.set_yticklabels(M_display, fontsize=7)
            if ri == nrows - 1:
                ax.set_xlabel("T", fontsize=8)
            for i in range(disp.shape[0]):
                for j in range(disp.shape[1]):
                    v = disp[i, j]
                    if np.isnan(v):
                        continue
                    rel   = (np.log(max(v, vmin)) - np.log(vmin)) / log_span
                    color = "white" if rel > 0.65 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color=color, fontsize=6)
    cbar = fig.colorbar(im, ax=list(axes.flat), fraction=0.02, pad=0.02)
    cbar.set_label("max / mean", fontsize=8)
    cbar.ax.yaxis.set_major_locator(FixedLocator([1, 2, 4, 8]))
    cbar.ax.yaxis.set_major_formatter(FixedFormatter(["1", "2", "4", "8"]))
    cbar.ax.yaxis.set_minor_locator(NullLocator())
    cbar.ax.tick_params(labelsize=7)
    plt.savefig(os.path.join(OUTPUT_DIR, "causal_load_imbalance_heatmap.png"),
                dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_speedup()
    plot_speedup_std()
    plot_profile()
    plot_causal_speedup()
    plot_causal_speedup_std()
    plot_steal_success_heatmap()
    plot_load_imbalance_heatmap()
    write_causal_summary()
