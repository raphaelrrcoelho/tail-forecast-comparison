#!/usr/bin/env python3
"""Figures and tables for main.tex from results/*.json.

  fig_detection.pdf  rounds to separate the Gaussian pair, against alpha
  fig_dm.pdf         DM rejections favouring the narrower interval at the
                     null, against the expected number of misses
  tab_null.tex       error rates under the null (Design N)
  tab_dmcorr.tex     small-sample corrections of DM under the null (sim_dm.py)
  tab_power.tex      power of DM, its bootstrap version and the tuned EB test (Design D)
  tab_empirical.tex  summary of the online conformal comparisons
  tab_es.tex         summary of the (VaR, ES) comparisons

Usage: python figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3
INK, MUTED = "#1f1f1e", "#8a8984"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "lines.linewidth": 2, "mathtext.fontset": "cm"})


def detection_figure():
    det = json.loads((RES / "sim_detect.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    for ax, sign, title in zip(axes, (-1, 1), ("A narrowed", "A widened")):
        rows = sorted((r for r in det if r["sign"] == sign), key=lambda r: r["alpha"])
        a = np.array([r["alpha"] for r in rows])
        eb = np.array([r["eb_median"] for r in rows])
        lo = np.array([r["eb_q10"] for r in rows])
        hi = np.array([r["eb_q90"] for r in rows])
        ho = np.array([r["hoeffding"] for r in rows], dtype=float)
        dm = np.array([r["fixed_T_80"] for r in rows])
        lb = np.array([r["lower_scale"] for r in rows])
        ax.fill_between(a, lo, hi, color=BLUE, alpha=0.18, linewidth=0)
        ax.loglog(a, eb, "-o", color=BLUE, ms=4, label="EB sequence (median)")
        ax.loglog(a, ho, "--s", color=ORANGE, ms=4, label="Hoeffding sequence")
        ax.loglog(a, dm, ":^", color=AQUA, ms=4, label="known-variance test, 80% power")
        ax.loglog(a, lb, "-", color=MUTED, lw=1, label=r"$H^2/(\alpha\Delta^2)$")
        ax.set_xlabel(r"miscoverage level $\alpha$")
        ax.set_title(title, fontsize=9, color=INK)
        ax.grid(which="major", color=MUTED, alpha=0.2, lw=0.5)
        ax.set_xticks(a)
        ax.set_xticklabels([f"{x:g}" for x in a])
        ax.minorticks_off()
        ax.invert_xaxis()
    axes[0].set_ylabel("rounds to separate $A$ from $B$")
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(HERE / "fig_detection.pdf")
    plt.close(fig)


def dm_figure():
    null = json.loads((RES / "sim_null.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    markers = {0.2: "o", 0.1: "s", 0.05: "^", 0.02: "D", 0.01: "v"}
    for ax, design, title in zip(axes, ("gauss", "t3"),
                                 ("Gaussian, constant scale", "Student-$t_3$, random scale")):
        for r in sorted((r for r in null if r["design"] == design), key=lambda r: -r["alpha"]):
            ax.semilogx(r["mean_misses_A"], r["dm_narrow"], markers[r["alpha"]], color=BLUE, ms=4.5,
                        mew=1.2, label=fr"$\alpha={r['alpha']}$", linestyle="none")
            ax.semilogx(r["mean_misses_A"], r["dm_wide"], markers[r["alpha"]], color=ORANGE, ms=4.5,
                        mfc="none", mew=1.2, linestyle="none")
        ax.axhline(0.025, color=MUTED, lw=1, ls="--")
        ax.set_xlabel("expected misses of the narrower interval, $Tp_A$")
        ax.set_title(title, fontsize=9, color=INK)
        ax.grid(which="major", color=MUTED, alpha=0.2, lw=0.5)
    axes[0].set_ylabel("rejection rate")
    axes[0].text(0.98, 0.95, "favours narrower (filled)\nfavours wider (open)\nnominal 2.5% (dashed)",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=7, color=INK)
    axes[1].legend(loc="upper right", fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(HERE / "fig_dm.pdf")
    plt.close(fig)


def pct(x):
    return f"{100 * x:.1f}"


def null_table():
    null = json.loads((RES / "sim_null.json").read_text())
    lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r" & & & \multicolumn{2}{c}{DM at $T$} & \multicolumn{3}{c}{error by $T$} & \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
             r"design & $\alpha$ & $T$ & narrower & wider & EB & Hoeffding & asympt. & $Tp_A$\\",
             r"\midrule"]
    for design, name in (("gauss", "Gaussian"), ("t3", "$t_3$")):
        for r in sorted((r for r in null if r["design"] == design), key=lambda r: -r["alpha"]):
            for j, T in enumerate(r["T"]):
                lines.append(f"{name if (j == 0 and r['alpha'] == 0.2) else ''} & "
                             f"{r['alpha'] if j == 0 else ''} & {T} & {pct(r['dm_narrow'][j])} & "
                             f"{pct(r['dm_wide'][j])} & {pct(r['eb_u'][j])} & {pct(r['hoef_u'][j])} & "
                             f"{pct(r['asym_u'][j])} & {r['mean_misses_A'][j]:.0f}\\\\")
        lines.append(r"\midrule" if design == "gauss" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    (HERE / "tab_null.tex").write_text("\n".join(lines) + "\n")


def dmcorr_table():
    res = json.loads((RES / "sim_dm.json").read_text())["null"]
    null = json.loads((RES / "sim_null.json").read_text())
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"design & $\alpha$ & $T$ & $Tp_A$ & DM & HLN & WPE & bootstrap\\", r"\midrule"]
    for design, name in (("gauss", "Gaussian"), ("t3", "$t_3$")):
        for r in sorted((r for r in res if r["design"] == design), key=lambda r: -r["alpha"]):
            n = next(x for x in null if x["design"] == design and x["alpha"] == r["alpha"])
            for j, T in enumerate(r["T"]):
                lines.append(f"{name if (j == 0 and r['alpha'] == 0.2) else ''} & "
                             f"{r['alpha'] if j == 0 else ''} & {T} & {n['mean_misses_A'][j]:.0f} & "
                             + " & ".join(pct(r[k + "_narrow"][j]) for k in ("dm", "hln", "wpe", "boot")) + "\\\\")
        lines.append(r"\midrule" if design == "gauss" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    (HERE / "tab_dmcorr.tex").write_text("\n".join(lines) + "\n")


def power_table():
    det = json.loads((RES / "sim_detect.json").read_text())
    boot = json.loads((RES / "sim_dm.json").read_text())["detect"]
    lines = [r"\setlength{\tabcolsep}{4pt}\begin{tabular}{rrrrrrrrrrrrrrr}", r"\toprule",
             r" & \multicolumn{7}{c}{$A$ narrowed} & \multicolumn{7}{c}{$A$ widened}\\",
             r"\cmidrule(lr){2-8}\cmidrule(lr){9-15}",
             r" & \multicolumn{3}{c}{$T=T_{80}$} & \multicolumn{3}{c}{$T=2T_{80}$} & $4T_{80}$"
             r" & \multicolumn{3}{c}{$T=T_{80}$} & \multicolumn{3}{c}{$T=2T_{80}$} & $4T_{80}$\\",
             r"$\alpha$" + (" & DM & boot & EB" * 2 + " & EB") * 2 + r"\\", r"\midrule"]
    for a in sorted({r["alpha"] for r in det}, reverse=True):
        cells = []
        for sign in (-1, 1):
            r = next(r for r in det if r["alpha"] == a and r["sign"] == sign)
            b = next(r for r in boot if r["alpha"] == a and r["sign"] == sign)
            for j in (0, 1):
                cells += [pct(r["power_dm"][j]), pct(b["power_boot"][j]), pct(r["power_ebt"][j])]
            cells.append(pct(r["power_ebt"][2]))
        lines.append(f"{a:g} & " + " & ".join(cells) + "\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "tab_power.tex").write_text("\n".join(lines) + "\n")


def empirical_table():
    emp = json.loads((RES / "empirical.json").read_text())
    S = emp["summary"]
    lines = [r"\begin{tabular}{rrrrrrrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{significant (\%)} & \multicolumn{4}{c}{first half}\\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-9}",
             r" & & & & & \multicolumn{2}{c}{DM} & \multicolumn{2}{c}{EB}\\",
             r"\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
             r"level & misses & DM & EB & Hoeffding & signif. & sign agrees (\%) & signif. & sign agrees (\%)\\", r"\midrule"]
    for a in sorted(S, key=lambda k: -float(k)):
        s = S[a]
        lines.append(f"{100 * (1 - float(a)):.0f}\\% & {s['n_inf_med']:.0f} & "
                     f"{pct(s['dm_sig'])} & {pct(s['ebt_sig'])} & {pct(s['hoef_sig'])} & "
                     f"{s['dm1_n']} & {pct(s['dm1_rep'])} & {s['ebt1_n']} & {pct(s['ebt1_rep'])}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "tab_empirical.tex").write_text("\n".join(lines) + "\n")


def es_table():
    base = json.loads((RES / "es.json").read_text())
    alt = {k: json.loads((RES / f"es_kappa{k}.json").read_text())["summary"] for k in ("1.5", "3")
           if (RES / f"es_kappa{k}.json").exists()}
    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             r" & $\tau=0.025$ & $\tau=0.01$\\", r"\midrule",
             r"\multicolumn{3}{l}{\emph{250-day windows}}\\"]
    S = base["summary"]
    row = lambda label, key, f: lines.append(f"{label} & {f(S['0.025'][key])} & {f(S['0.01'][key])}\\\\")
    row("window--pair tests", "windows", lambda x: f"{x}")
    row("median exceedances", "win_exc_med", lambda x: f"{x:.0f}")
    row("DM significant (\\%)", "win_dm_sig", pct)
    row("\\quad favouring the less conservative (\\%)", "win_sig_favours_less", pct)
    row("\\quad wrong sign, favouring less conservative (\\%)", "win_wrong_sign_less_rest", pct)
    row("\\quad wrong sign, favouring more conservative (\\%)", "win_wrong_sign_more_rest", pct)
    row("EB certifies (\\%)", "win_eb_any", pct)
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{3}{l}{\emph{full period}}\\")
    row("pairs", "pairs", lambda x: f"{x}")
    row("median exceedances", "full_exc_med", lambda x: f"{x:.0f}")
    row("DM significant (\\%)", "full_dm_sig", pct)
    row("\\quad favouring the less conservative (\\%)", "full_sig_favours_less", pct)
    row("EB certifies (\\%)", "full_ebt_any", pct)
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "tab_es.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    for f, need in ((detection_figure, "sim_detect.json"), (dm_figure, "sim_null.json"),
                    (null_table, "sim_null.json"), (dmcorr_table, "sim_dm.json"),
                    (power_table, "sim_detect.json"), (empirical_table, "empirical.json"),
                    (es_table, "es.json")):
        if (RES / need).exists():
            f()
            print("done:", f.__name__)
        else:
            print("skipped (no results):", f.__name__)
