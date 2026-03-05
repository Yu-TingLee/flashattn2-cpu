import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FixedLocator

OUTPUT_DIR = os.path.join("outputs", "plots")

LINEWIDTH  = 1.5
GRID_ALPHA = 0.7
X_TICKS    = [32, 128, 256, 384, 512, 640, 768, 896, 1024]
X_LIMITS   = (32, 1024)

IMPL_STYLES = {
    'jit':        {'label': 'FA-2 (Numba)',            'color': None},
    'fa2':        {'label': 'FA-2 (NumPy)',             'color': 'green'},
    'fa2_cpp':    {'label': 'FA-2 (C++, ST)',           'color': 'purple'},
    'fa2_cpp_mt': {'label': 'FA-2 (C++, MT)',           'color': 'yellow'},
    'naive_py':   {'label': 'Naive Attention (Python)', 'color': 'r',      'linestyle': '--'},
    'naive_cpp':  {'label': 'Naive Attention (C++)',    'color': 'orange', 'linestyle': '--'},
}

PROFILE_SEGMENTS = ["line7", "line8", "line10", "line11", "others"]
PROFILE_LABELS   = {"line7": "Line 7",  "line8": "Line 8",  "line10": "Line 10",
                    "line11": "Line 11", "others": "Others"}
PROFILE_COLORS   = {"line7": "#4C72B0", "line8": "#55A868", "line10": "#C44E52",
                    "line11": "#8172B3", "others": "#CCB974"}


def get_smart_ticks(y_data, candidates):
    if not y_data:
        return candidates
    y_min, y_max = min(min(y_data), 90), max(y_data)
    low_tick  = next((c for c in reversed(candidates) if c <= y_min), candidates[0])
    high_tick = next((c for c in candidates if c > y_max), candidates[-1])
    if y_max >= candidates[-1]:
        high_tick = candidates[-1]
    return [c for c in candidates if low_tick <= c <= high_tick]


def load_runtime_data():
    df_naive  = pd.read_csv(os.path.join("outputs", "naive_attn",        "runtime.csv"))
    df_cpp_o3 = pd.read_csv(os.path.join("outputs", "naive_attn_cpp_O3", "runtime.csv"))

    def load_and_agg(target_name, col_name):
        df     = pd.read_csv(os.path.join("outputs", target_name, "runtime.csv"))
        merged = pd.merge(df, df_cpp_o3, on=["testset", "T", "d"], suffixes=("", "_cpp"))
        merged[col_name] = (merged["runtime"] / merged["runtime_cpp"]) * 100
        return merged.groupby(["T", "d", "M_bytes"])[col_name].mean().reset_index()

    agg_jit        = load_and_agg("fa2_jit",      "pct_runtime_jit")
    agg_fa2        = load_and_agg("fa2",           "pct_runtime_fa2")
    agg_fa2_cpp    = load_and_agg("fa2_cpp_O3",    "pct_runtime_fa2_cpp")
    agg_fa2_cpp_mt = load_and_agg("fa2_cpp_O3_mt", "pct_runtime_fa2_cpp_mt")

    return df_naive, df_cpp_o3, agg_jit, agg_fa2, agg_fa2_cpp, agg_fa2_cpp_mt


def get_unique_values(dfs, col):
    vals = set()
    for df in dfs:
        if not df.empty:
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


def add_split_legend(fig, handles, labels, y_pos_top=0.03, y_pos_bottom=0.01):
    kwargs = dict(fontsize=7, loc='lower center', columnspacing=0.6,
                  handlelength=1.5, frameon=False)
    if len(handles) == 5:
        fig.legend(handles[:3], labels[:3], bbox_to_anchor=(0.52, y_pos_top),    ncol=3, **kwargs)
        fig.legend(handles[3:], labels[3:], bbox_to_anchor=(0.52, y_pos_bottom), ncol=2, **kwargs)
    else:
        ncol = min(len(labels), 3) if labels else 1
        fig.legend(handles, labels, bbox_to_anchor=(0.52, y_pos_top), ncol=ncol, **kwargs)


def plot_runtime_subplot(ax, agg_jit, agg_fa2, agg_fa2_cpp, agg_fa2_cpp_mt, df_cpp_o3, df_naive,
                         t_val, d_val, plot_flags, is_speedup=False):
    series = [
        ('jit',        agg_jit,        'pct_runtime_jit'),
        ('fa2',        agg_fa2,        'pct_runtime_fa2'),
        ('fa2_cpp',    agg_fa2_cpp,    'pct_runtime_fa2_cpp'),
        ('fa2_cpp_mt', agg_fa2_cpp_mt, 'pct_runtime_fa2_cpp_mt'),
    ]
    cpp_o3 = filter_data(df_cpp_o3, t_val, d_val)
    py_sub = filter_data(df_naive,  t_val, d_val)
    y_data = [1] if is_speedup else [100]

    for key, agg_df, col in series:
        s = filter_data(agg_df, t_val, d_val).sort_values("M_bytes")
        if not plot_flags[key] or s.empty:
            continue
        y      = 100.0 / s[col] if is_speedup else s[col]
        style  = IMPL_STYLES[key]
        kwargs = {'linewidth': LINEWIDTH, 'label': style['label']}
        if style.get('color'):
            kwargs['color'] = style['color']
        ax.plot(s["M_bytes"] / 1024, y, **kwargs)
        y_data.extend(y)

    if plot_flags['naive_py'] and not cpp_o3.empty and not py_sub.empty:
        style = IMPL_STYLES['naive_py']
        val   = (cpp_o3["runtime"].mean() / py_sub["runtime"].mean()) if is_speedup \
                else (py_sub["runtime"].mean() / cpp_o3["runtime"].mean()) * 100
        ax.axhline(y=val, color=style['color'], linestyle=style['linestyle'],
                   linewidth=LINEWIDTH, label=style['label'])
        y_data.append(val)

    if plot_flags['naive_cpp']:
        style = IMPL_STYLES['naive_cpp']
        ax.axhline(y=1 if is_speedup else 100,
                   color=style['color'], linestyle=style['linestyle'],
                   linewidth=LINEWIDTH, label=style['label'])

    return y_data


def set_y_ticks(ax, y_data, is_speedup=False):
    if is_speedup:
        candidates, suffix = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 150], 'x'
    else:
        candidates, suffix = [0.5, 1, 5, 25, 50, 100, 200, 400, 800], '%'
    ticks = get_smart_ticks(y_data, candidates)
    ax.set_ylim(min(ticks), max(ticks))
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{t}{suffix}' for t in ticks])


def _fill_subplots(axes, nrows, T_values, d_values, subplot_args, flags, is_speedup, logscale):
    legend_ax = None
    for row, t_val in enumerate(T_values):
        for col, d_val in enumerate(d_values):
            ax     = axes[row, col] if nrows > 1 else axes[col]
            y_data = plot_runtime_subplot(ax, *subplot_args, t_val, d_val, flags, is_speedup)
            setup_axis(ax, t_val, d_val, logscale)
            set_y_ticks(ax, y_data, is_speedup)
            if legend_ax is None:
                legend_ax = ax
    return legend_ax


def plot_performance(
    plot_jit=True, plot_fa2=True, plot_fa2_cpp=True, plot_fa2_cpp_mt=True,
    plot_naive_py=True, plot_naive_cpp=True,
    also_plot_speedup=True,
    plot_jit_speedup=None, plot_fa2_speedup=None, plot_fa2_cpp_speedup=None,
    plot_fa2_cpp_mt_speedup=None, plot_naive_py_speedup=None, plot_naive_cpp_speedup=None,
    y_logscale=False, y_logscale_speedup=False
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        df_naive, df_cpp_o3, agg_jit, agg_fa2, agg_fa2_cpp, agg_fa2_cpp_mt = load_runtime_data()
    except FileNotFoundError as e:
        print(f"Error: CSV missing. {e}")
        return

    T_values = get_unique_values([agg_jit, agg_fa2, agg_fa2_cpp, agg_fa2_cpp_mt], "T")
    d_values = get_unique_values([agg_jit, agg_fa2, agg_fa2_cpp, agg_fa2_cpp_mt], "d")
    nrows, ncols = len(T_values), len(d_values)

    plot_flags = {
        'jit': plot_jit, 'fa2': plot_fa2, 'fa2_cpp': plot_fa2_cpp,
        'fa2_cpp_mt': plot_fa2_cpp_mt, 'naive_py': plot_naive_py, 'naive_cpp': plot_naive_cpp,
    }
    speedup_flags = {
        'jit':        plot_jit        if plot_jit_speedup        is None else plot_jit_speedup,
        'fa2':        plot_fa2        if plot_fa2_speedup        is None else plot_fa2_speedup,
        'fa2_cpp':    plot_fa2_cpp    if plot_fa2_cpp_speedup    is None else plot_fa2_cpp_speedup,
        'fa2_cpp_mt': plot_fa2_cpp_mt if plot_fa2_cpp_mt_speedup is None else plot_fa2_cpp_mt_speedup,
        'naive_py':   plot_naive_py,
        'naive_cpp':  plot_naive_cpp  if plot_naive_cpp_speedup  is None else plot_naive_cpp_speedup,
    }

    subplot_args = (agg_jit, agg_fa2, agg_fa2_cpp, agg_fa2_cpp_mt, df_cpp_o3, df_naive)

    # Relative runtime plot
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5, nrows * 2), sharex=True, sharey=False)
    legend_ax = _fill_subplots(axes, nrows, T_values, d_values, subplot_args,
                                plot_flags, is_speedup=False, logscale=y_logscale)
    fig.text(0.52, 0.07, "Cache Budget M (KB)",  ha='center', va='center', fontsize=9)
    fig.text(0.02, 0.5,  "Relative Runtime (%)", ha='center', va='center', rotation='vertical', fontsize=9)
    if legend_ax:
        h, l = legend_ax.get_legend_handles_labels()
        add_split_legend(fig, h, l)
    plt.tight_layout(rect=[0.04, 0.08, 1, 0.98])
    plt.savefig(os.path.join(OUTPUT_DIR, "relative_runtime.png"), dpi=600)
    plt.close()

    # Speedup plot
    if also_plot_speedup:
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(6.5, nrows * 2), sharex=True, sharey=False)
        legend_ax2  = _fill_subplots(axes2, nrows, T_values, d_values, subplot_args,
                                      speedup_flags, is_speedup=True, logscale=y_logscale_speedup)
        fig2.text(0.52, 0.07, "Cache Budget M (KiB)",     ha='center', va='center', fontsize=9)
        fig2.text(0.02, 0.5,  "Speedup (x-times faster)", ha='center', va='center', rotation='vertical', fontsize=9)
        if legend_ax2:
            h, l = legend_ax2.get_legend_handles_labels()
            add_split_legend(fig2, h, l)
        plt.tight_layout(rect=[0.04, 0.08, 1, 0.98])
        plt.savefig(os.path.join(OUTPUT_DIR, "speedup_xtimes.png"), dpi=600)
        plt.close()

    _write_summary(agg_jit, agg_fa2, agg_fa2_cpp, agg_fa2_cpp_mt)
    _plot_combined(agg_jit, agg_fa2, agg_fa2_cpp, agg_fa2_cpp_mt, df_cpp_o3, df_naive,
                   T_values, d_values, plot_flags, speedup_flags, y_logscale, y_logscale_speedup)


def _write_summary(agg_jit, agg_fa2, agg_fa2_cpp, agg_fa2_cpp_mt):
    summary_lines = []
    for label, agg_df, col in [
        ("Numba",       agg_jit,        "pct_runtime_jit"),
        ("FA2 (NumPy)", agg_fa2,        "pct_runtime_fa2"),
        ("FA2 (C++ ST)",agg_fa2_cpp,    "pct_runtime_fa2_cpp"),
        ("FA2 (C++ MT)",agg_fa2_cpp_mt, "pct_runtime_fa2_cpp_mt"),
    ]:
        top    = agg_df.nsmallest(10, col)
        bottom = agg_df.nlargest(10, col)
        summary_lines.append(f"Top {len(top)} Best Performance Boosts ({label}):")
        for _, row in top.iterrows():
            speedup = 100.0 / row[col] if row[col] > 0 else float('inf')
            summary_lines.append(
                f"  T={int(row['T'])}, d={int(row['d'])}, M={int(row['M_bytes']/1024)}KB, "
                f"Runtime: {row[col]:.2f}%, Speedup: {speedup:.2f}x"
            )
        summary_lines.append(f"\nTop {len(bottom)} Least Performance Boosts ({label}):")
        for _, row in bottom.iterrows():
            speedup = 100.0 / row[col] if row[col] > 0 else float('inf')
            summary_lines.append(
                f"  T={int(row['T'])}, d={int(row['d'])}, M={int(row['M_bytes']/1024)}KB, "
                f"Runtime: {row[col]:.2f}%, Speedup: {speedup:.2f}x"
            )
        summary_lines.append("-" * 30)

    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))


def _plot_combined(agg_jit, agg_fa2, agg_fa2_cpp, agg_fa2_cpp_mt, df_cpp_o3, df_naive,
                   T_values, d_values, plot_flags, speedup_flags, y_logscale, y_logscale_speedup):
    nrows, ncols  = len(T_values), len(d_values)
    fig, axes     = plt.subplots(nrows, ncols * 2, figsize=(13, nrows * 2), sharex=True, sharey=False)
    subplot_args  = (agg_jit, agg_fa2, agg_fa2_cpp, agg_fa2_cpp_mt, df_cpp_o3, df_naive)

    left  = axes[:, :ncols] if nrows > 1 else axes[:ncols]
    right = axes[:, ncols:] if nrows > 1 else axes[ncols:]

    legend_ax = _fill_subplots(left,  nrows, T_values, d_values, subplot_args,
                                plot_flags,    is_speedup=False, logscale=y_logscale)
    _fill_subplots(            right, nrows, T_values, d_values, subplot_args,
                                speedup_flags, is_speedup=True,  logscale=y_logscale_speedup)

    fig.text(0.515, 0.045, "Cache Budget M (KiB)",      ha='center', va='center', fontsize=14)
    fig.text(0.015, 0.5,   "Relative Runtime (%)",      ha='center', va='center', rotation='vertical', fontsize=14)
    fig.text(0.512, 0.5,   "Speedup (x-times faster)",  ha='center', va='center', rotation='vertical', fontsize=14)

    if legend_ax:
        h, l = legend_ax.get_legend_handles_labels()
        fig.legend(h, l, fontsize=11, loc='lower center', bbox_to_anchor=(0.51, 0),
                   ncol=len(h), columnspacing=0.8, handlelength=1.5, frameon=False)

    plt.tight_layout(rect=[0.02, 0.06, 1, 0.98], w_pad=2.5)
    plt.savefig(os.path.join(OUTPUT_DIR, "combined_runtime_speedup.png"), dpi=600)
    plt.close()


def _draw_pie(ax, row_data, pct_label, fontsize=6, radius=1.0):
    values = [max(float(row_data.iloc[0][s]), 0.0) for s in PROFILE_SEGMENTS]
    ax.pie(values,
           colors=[PROFILE_COLORS[s] for s in PROFILE_SEGMENTS],
           startangle=90, autopct=pct_label, pctdistance=0.65,
           wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
           textprops={'fontsize': fontsize, 'fontweight': 'bold', 'color': 'white'},
           radius=radius)
    ax.axis('equal')


def plot_profile_breakdown(opt_flag="O3", include_python=True, include_cpp=True):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    impl_frames, impl_rows = [], []

    def load_profile(path, label):
        df = pd.read_csv(path)
        df["impl"] = label
        return df

    if include_python:
        path_py = os.path.join("outputs", "fa2_profile", "runtime.csv")
        if os.path.exists(path_py):
            impl_frames.append(load_profile(path_py, "FlashAttention-2 (Python)"))
            impl_rows.append("FlashAttention-2 (Python)")
        else:
            print(f"Warning: missing profile CSV {path_py}")

    if include_cpp:
        path_cpp = os.path.join("outputs", f"fa2_cpp_profile_{opt_flag}", "runtime.csv")
        if os.path.exists(path_cpp):
            impl_frames.append(load_profile(path_cpp, "FlashAttention-2 (C++)"))
            impl_rows.append("FlashAttention-2 (C++)")
        else:
            print(f"Warning: missing profile CSV {path_cpp}")

    if not impl_frames:
        print("No profile data available.")
        return

    profile_df = pd.concat(impl_frames, ignore_index=True)
    agg = profile_df.groupby(["T", "M_bytes", "impl"], as_index=False)[PROFILE_SEGMENTS].mean()
    if agg.empty:
        print("Profile data is empty after aggregation.")
        return
    agg = agg.sort_values(["T", "M_bytes"])

    segment_handles = [mpatches.Patch(facecolor=PROFILE_COLORS[s], label=PROFILE_LABELS[s])
                       for s in PROFILE_SEGMENTS]
    legend_labels   = [h.get_label() for h in segment_handles]
    t_values        = sorted(agg["T"].unique())
    max_cols        = 3

    def pct_label(pct):
        return f"{pct:.0f}%" if pct >= 5 else ""

    for t_val in t_values:
        subset   = agg[agg["T"] == t_val]
        if subset.empty:
            continue
        unique_m = sorted(subset["M_bytes"].unique())[:max_cols]

        fig, axes = plt.subplots(len(impl_rows), 3, figsize=(3.5, len(impl_rows) * 1.6))
        axes = np.atleast_2d(axes)
        fig.patch.set_facecolor('white')

        for row_idx, impl in enumerate(impl_rows):
            impl_short = "Python" if "Python" in impl else "C++"
            for col_idx in range(3):
                ax = axes[row_idx, col_idx]
                ax.set_facecolor('white')
                if col_idx < len(unique_m):
                    m_val    = unique_m[col_idx]
                    row_data = subset[(subset["impl"] == impl) & (subset["M_bytes"] == m_val)]
                    _draw_pie(ax, row_data, pct_label, fontsize=6)
                    if row_idx == 0:
                        ax.set_title(f"M={int(m_val/1024)} KiB", fontsize=8, pad=2)
                else:
                    ax.axis('off')
                if col_idx == 0:
                    ax.set_ylabel(impl_short, fontsize=8, labelpad=8)

        fig.text(0.52, 0.97, f"T={int(t_val)}", ha='center', va='center', fontsize=9)
        if t_val == max(t_values):
            fig.legend(segment_handles, legend_labels, loc='lower center',
                       bbox_to_anchor=(0.52, 0.01), ncol=len(segment_handles),
                       fontsize=7, frameon=False, handlelength=1.2, columnspacing=0.8)
            plt.tight_layout(rect=[0.04, 0.06, 0.99, 0.93], pad=0.3, w_pad=0.1, h_pad=0.3)
        else:
            plt.tight_layout(rect=[0.04, 0, 0.99, 0.93], pad=0.3, w_pad=0.1)
        plt.savefig(os.path.join(OUTPUT_DIR, f"profile_T{int(t_val)}.png"), dpi=600,
                    bbox_inches='tight', pad_inches=0.02)
        plt.close()

    _plot_combined_profile(agg, t_values, impl_rows, segment_handles, legend_labels,
                           max_cols, pct_label)


def _plot_combined_profile(agg, t_values, impl_rows, segment_handles, legend_labels,
                           max_cols, pct_label):
    n_t          = len(t_values)
    n_impl       = len(impl_rows)
    n_cols_total = n_t * max_cols

    fig, axes = plt.subplots(n_impl, n_cols_total, figsize=(13, n_impl * 2.2))
    axes = np.atleast_2d(axes)
    fig.patch.set_facecolor('white')

    for t_idx, t_val in enumerate(t_values):
        subset     = agg[agg["T"] == t_val]
        if subset.empty:
            continue
        unique_m   = sorted(subset["M_bytes"].unique())[:max_cols]
        col_offset = t_idx * max_cols

        for impl_idx, impl in enumerate(impl_rows):
            impl_short = "Python" if "Python" in impl else "C++"
            for m_idx in range(max_cols):
                global_col = col_offset + m_idx
                ax = axes[impl_idx, global_col]
                ax.set_facecolor('white')
                if m_idx < len(unique_m):
                    m_val    = unique_m[m_idx]
                    row_data = subset[(subset["impl"] == impl) & (subset["M_bytes"] == m_val)]
                    _draw_pie(ax, row_data, pct_label, fontsize=9, radius=1.3)
                    if impl_idx == 0:
                        ax.set_title(f"M={int(m_val/1024)} KiB", fontsize=10, pad=4)
                else:
                    ax.axis('off')
                if global_col == 0:
                    ax.set_ylabel(impl_short, fontsize=10, labelpad=12)

    plt.subplots_adjust(left=0.04, right=0.99, top=0.82, bottom=0.14, wspace=0.05, hspace=0.0)
    fig.canvas.draw()

    for t_idx, t_val in enumerate(t_values):
        col_offset = t_idx * max_cols
        bbox_left  = axes[0, col_offset].get_position()
        bbox_right = axes[0, col_offset + max_cols - 1].get_position()
        center_x   = (bbox_left.x0 + bbox_right.x1) / 2
        fig.text(center_x - 0.003, 0.91, f"T={int(t_val)}", ha='center', va='center', fontsize=12)
        if t_idx < n_t - 1:
            bbox_next = axes[0, col_offset + max_cols].get_position()
            sep_x     = (bbox_right.x1 + bbox_next.x0) / 2
            fig.add_artist(plt.Line2D([sep_x, sep_x], [0.16, 0.82],
                                      transform=fig.transFigure, color='gray', linewidth=1.5))

    fig.legend(segment_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.52, 0.01),
               ncol=len(segment_handles), fontsize=10, frameon=False,
               handlelength=1.8, columnspacing=1.2)

    plt.savefig(os.path.join(OUTPUT_DIR, "profile.png"), dpi=600, bbox_inches='tight', pad_inches=0.03)
    plt.close()


if __name__ == "__main__":
    plot_performance(y_logscale=True, y_logscale_speedup=True)
    plot_profile_breakdown()
