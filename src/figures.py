"""Paper figures and tables from results/*.json. Deterministic, no data re-analysis."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

from common import RESULTS

FIGS = RESULTS.parent / "figures"
FIGS.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "figure.dpi": 200})


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def fig_accuracy_vs_L(ev):
    n50 = ev["by_n"]["50"]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    curves = [("testA_markov", n50["testA_markov"], "o-", "Markov (k=2)"),
              ("lstm", ev.get("testA_lstm_N50", {}), "s-", "LSTM")]
    for _, cur, style, label in curves:
        ls = sorted(int(L) for L, v in cur.items() if v)
        accs = [cur[str(L)]["acc"] for L in ls]
        ax.plot(ls, accs, style, label=label, ms=3)
    ls = sorted(int(L) for L, v in n50["testA_markov"].items() if v)
    band = [n50["testA_markov"][str(L)]["null_q95"] for L in ls]
    ax.fill_between(ls, 0, band, alpha=0.25, color="gray",
                    label="permutation 95% band")
    surr = ev.get("surrogate_control_L48_N50")
    if surr:
        ax.errorbar([48], [surr["acc_mean"]], yerr=[2 * surr["acc_sd"]],
                    fmt="d", color="crimson", label="AAFT surrogate (L=48)", ms=4)
    ml = n50.get("testA_meanlevel_L48")
    if ml:
        ax.plot([48], [ml["acc"]], "^", color="darkorange",
                label="mean-level baseline (L=48)", ms=4)
    ax.axhline(n50["chance"], color="k", lw=0.6, ls=":")
    ax.set_xlabel("sequence length L (symbols)")
    ax.set_ylabel("identification accuracy")
    ax.set_xscale("log")
    ax.set_xticks(ls)
    ax.set_xticklabels(ls)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_accuracy_vs_L.pdf")


def fig_accuracy_vs_N(ev):
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ns = sorted(int(n) for n in ev["by_n"])
    accs = [ev["by_n"][str(n)]["testA_markov"]["48"]["acc"] for n in ns]
    chance = [1.0 / n for n in ns]
    ax.plot(ns, accs, "o-", label="Markov accuracy (L=48)", ms=3)
    ax.plot(ns, chance, "k:", label="chance 1/N")
    ax.set_xlabel("number of elements N")
    ax.set_ylabel("accuracy")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels(ns)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_accuracy_vs_N.pdf")


def fig_drift(dr):
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    stride, win = dr["stride_h"], dr["window_h"]
    labels = {"sym_loglik": "symbolic: log-lik", "sym_jsd": "symbolic: JSD",
              "raw_mean": "raw: mean", "raw_var": "raw: variance"}
    colors = {"sym_loglik": "tab:blue", "sym_jsd": "tab:cyan",
              "raw_mean": "tab:orange", "raw_var": "tab:red"}
    for nm, sig in dr["signals_test"].items():
        t = np.arange(len(sig)) * stride + win  # hours after Dec 13 00:00
        ax.plot(t / 24.0, sig, lw=0.9, label=labels[nm], color=colors[nm])
        alarm = dr["detectors"]["cusum"][nm]["alarm_hours_after_dec13"]
        if alarm is not None:
            ax.axvline(alarm / 24.0, color=colors[nm], ls="--", lw=0.7)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 * 1.28)  # headroom for legend + annotations
    ax.axvline(8, color="k", lw=0.8)  # Dec 21 = Test-A/Test-B boundary
    ax.text(8.15, y0 + 0.04 * (y1 - y0), "Test-B\nstarts", fontsize=5.5, va="bottom")
    ax.axvline(12, color="gray", lw=0.8, ls=":")  # Dec 25
    ax.text(12.15, y0 + 0.04 * (y1 - y0), "Dec 25", fontsize=5.5, va="bottom")
    ax.set_xlabel("days after 2013-12-13")
    ax.set_ylabel(r"aggregated $|z|$")
    ax.legend(fontsize=5.5, frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_drift.pdf")


def table_sweep(sw):
    lines = ["a  w   k=1    k=2    k=3"]
    grid = {}
    for r in sw["grid"]:
        grid[(r["alphabet"], r["paa_w"], r["order"])] = r["val_acc_select"]
    for a in (4, 6, 8):
        for w in (3, 6, 12):
            row = f"{a}  {w:>2}"
            for k in (1, 2, 3):
                row += f"  {grid[(a, w, k)]:.3f}"
            lines.append(row)
    (FIGS / "table_sweep.txt").write_text("\n".join(lines))


def main():
    ev, dr, sw = load("test_evaluation.json"), load("h3_drift.json"), load("sweep.json")
    if sw:
        table_sweep(sw)
    if ev:
        fig_accuracy_vs_L(ev)
        fig_accuracy_vs_N(ev)
    if dr:
        fig_drift(dr)
    print("figures written to", FIGS)


if __name__ == "__main__":
    main()
