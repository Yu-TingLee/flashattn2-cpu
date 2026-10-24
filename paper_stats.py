"""Prints every number quoted in the paper's results. Run from the repo root."""

import numpy as np
import pandas as pd

from plot import SCHEDULES, best, fa2_mt, naive, profile_shares, sdpa

pd.set_option("display.width", 200)


def summary(s, label):
    s = s.dropna()
    print(
        f"  {label:<34} n={len(s):2d}  min={s.min():.2f}  max={s.max():.2f}  "
        f"median={s.median():.2f}  mean={s.mean():.2f}  "
        f"geomean={np.exp(np.log(s).mean()):.2f}"
    )


def long_t(index):
    return index.get_level_values("T") >= 2048


def vs_sdpa():
    """FA-2 against SDPA and naive attention; returns the best runtimes per case."""
    res = {}
    for case, causal in (("self", False), ("masked", True)):
        nv = best(naive(causal)).runtime
        fa2 = best(fa2_mt(causal)).runtime
        sd = best(sdpa(causal)).runtime
        res[case] = {"naive": nv, "fa2": fa2, "sdpa": sd}
        big = long_t(fa2.index)
        print(f"\n===== {case.upper()} =====")
        summary(sd / fa2, "SDPA/FA-2 all T")
        summary((sd / fa2)[big], "SDPA/FA-2 T>=2K")
        summary((sd / fa2)[~big], "SDPA/FA-2 T<=1K")
        summary(nv / fa2, "naive/FA-2")
        summary((nv / fa2)[big], "naive/FA-2 T>=2K")
        print(
            pd.DataFrame(
                {
                    "naive_ms": nv * 1e3,
                    "fa2_ms": fa2 * 1e3,
                    "sdpa_ms": sd * 1e3,
                    "sdpa/fa2": sd / fa2,
                    "naive/fa2": nv / fa2,
                }
            )
            .round(3)
            .to_string()
        )
    return res


def masked_vs_self(res):
    print("\n===== MASKED / SELF (time ratio; ideal 0.5) =====")
    ms = pd.DataFrame(
        {
            f"{k}_masked/self": res["masked"][k] / res["self"][k]
            for k in ("fa2", "sdpa", "naive")
        }
    )
    print(ms.round(3).to_string())
    for c in ms:
        summary(ms[c][long_t(ms.index)], f"{c} T>=2K")


def schedules():
    print("\n===== SCHEDULES (masked, fixed threads; best M per schedule) =====")
    rows = []
    for nt in (8, 16):
        for s in SCHEDULES:
            b = best([f"fa2_causal_cpp_mt_{s}_t{nt}"], extra=("load_imbalance",))
            b["threads"], b["sched"] = nt, s
            rows.append(b)
    sch = pd.concat(rows).reset_index()
    rt = sch.pivot_table(index=["threads", "T", "d"], columns="sched", values="runtime")
    imb = sch.pivot_table(
        index=["threads", "T", "d"], columns="sched", values="load_imbalance"
    )
    out = pd.DataFrame(
        {
            "static_ms": rt.static * 1e3,
            "dyn_ms": rt.dynamic * 1e3,
            "ws_ms": rt.work_stealing * 1e3,
            "static/dyn": rt.static / rt.dynamic,
            "static/ws": rt.static / rt.work_stealing,
            "imb_static": imb.static,
            "imb_dyn": imb.dynamic,
            "imb_ws": imb.work_stealing,
        }
    )
    print(out.round(3).to_string())
    for nt in (8, 16):
        o = out.xs(nt, level="threads")
        big = long_t(o.index)
        print(f" threads={nt}")
        summary(o["static/dyn"], "static/dynamic all T")
        summary(o["static/ws"], "static/work-stealing all T")
        summary(o["static/ws"][big], "static/work-stealing T>=2K")
        summary(o.imb_static[big], "imbalance static T>=2K")
        summary(o.imb_dyn[big], "imbalance dynamic T>=2K")
        summary(o.imb_ws[big], "imbalance ws T>=2K")
        sd = best(sdpa(True, (nt,))).runtime
        for s in SCHEDULES:
            summary(sd / rt[s].xs(nt, level="threads"), f"SDPA/FA-2 {s} (same thr)")


def profile(T=8192, M=32768):
    print(f"\n===== MT PROFILE (% of thread time, T={T}, M={M // 1024} KiB) =====")
    at = profile_shares().xs((T, M), level=("T", "M_bytes"))
    print(at.round(1).to_string())
    print(" mean over 8 and 16 threads (as plotted)")
    print(at.groupby(["kernel", "d"]).mean().round(1).to_string())


if __name__ == "__main__":
    masked_vs_self(vs_sdpa())
    schedules()
    profile()
