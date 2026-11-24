"""Check Xenium non-malignant cell types against FISH, and their EGFR status.

Joins ``cells.csv`` from ``fish_to_xenium_cells.py`` with a cell-type table
(``cell_id,group``) and asks, per cell:

- Xenium call: non-malignant (normal cell types) or malignant (tumour states).
- FISH call: normal chromosome 7 (exactly ``--normal-cep7`` CEP7 spots in the
  nucleus, 2 by default) or abnormal (any other count).
- EGFR status from the FISH spots: amplified, gained or normal.

and reports how often the two calls agree, why they disagree (the CEP7 count),
and how many cells in each agreement group carry EGFR abnormalities.

Outputs next to ``cells.csv`` unless ``--output-dir`` is given:
``nonmalignant_cells.csv``, ``nonmalignant_summary.csv``,
``nonmalignant_report.html`` and ``.pdf``.

Run:
    uv run --no-sync python examples/registration/nonmalignant_fish_check.py \
        --cells outputs/.../refine/cells/cells.csv \
        --celltypes /path/to/annotations/p46p_celltype.csv
"""

from __future__ import annotations

import argparse
import base64
import io
import textwrap
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import chi2_contingency, mannwhitneyu, rankdata

NON_MALIGNANT = ["Endothelial", "Pericyte", "Myeloid", "Neuron", "Oligodendrocyte", "T cell"]
MALIGNANT = ["AC-like", "MES-like", "OPC-like", "Cycling"]
EGFR_ORDER = ["normal", "gain", "amplified"]
EGFR_COLORS = {"normal": "#b8c0c6", "gain": "#e0a33a", "amplified": "#c0392b"}
CEP7_COLORS = ["#d9dde0", "#9fb3c2", "#2e7d5b", "#e0a33a", "#c0392b"]
INK = "#1c2328"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", type=Path, required=True, help="cells.csv from fish_to_xenium_cells.py")
    p.add_argument("--celltypes", type=Path, required=True, help="CSV with columns cell_id,group")
    p.add_argument("--non-malignant", nargs="+", default=NON_MALIGNANT, help="Groups counted as non-malignant.")
    p.add_argument("--malignant", nargs="+", default=MALIGNANT,
                   help="Groups counted as malignant. 'Cycling' is included by default; drop it if you treat it as mixed.")
    p.add_argument("--normal-cep7", type=int, default=2, help="CEP7 spot count that counts as a normal FISH nucleus.")
    p.add_argument("--amp-ratio", type=float, default=2.0, help="EGFR / CEP7 spot ratio counted as amplified ...")
    p.add_argument("--amp-min-egfr", type=int, default=3, help="... with at least this many EGFR spots.")
    p.add_argument("--amp-min-egfr-no-cep7", type=int, default=4, help="EGFR spots counted as amplified when there is no CEP7 spot.")
    p.add_argument("--gain-egfr", type=int, default=3, help="Not amplified but at least this many EGFR spots: EGFR gain.")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def load(args: argparse.Namespace) -> pd.DataFrame:
    cells = pd.read_csv(args.cells)
    keep = (cells["nucleus_area_fish_px"] > 0) & cells["in_fish"] & ~cells["saturated"]
    if "debris" in cells:
        keep &= ~cells["debris"]
    cells = cells[keep]
    types = pd.read_csv(args.celltypes, usecols=["cell_id", "group"])
    cells = cells.merge(types, on="cell_id", how="inner")
    cells["xenium_call"] = np.where(cells["group"].isin(args.non_malignant), "non-malignant",
                                    np.where(cells["group"].isin(args.malignant), "malignant", "unused"))
    cells = cells[cells["xenium_call"] != "unused"].copy()

    cep7, egfr = cells["cep7_spots_nuc"], cells["egfr_spots_nuc"]
    cells["fish_call"] = np.where(cep7 == args.normal_cep7, f"CEP7 = {args.normal_cep7}", f"CEP7 != {args.normal_cep7}")
    cells["agreement"] = np.where(
        (cells["xenium_call"] == "non-malignant") == (cep7 == args.normal_cep7), "agree", "disagree")
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = egfr / cep7.where(cep7 > 0)
    amplified = ((cep7 > 0) & (ratio >= args.amp_ratio) & (egfr >= args.amp_min_egfr)) | (
        (cep7 == 0) & (egfr >= args.amp_min_egfr_no_cep7))
    cells["egfr_status"] = pd.Categorical(
        np.where(amplified, "amplified", np.where(egfr >= args.gain_egfr, "gain", "normal")), categories=EGFR_ORDER)
    cells["cep7_class"] = cep7.clip(upper=4).map({0: "0", 1: "1", 2: "2", 3: "3", 4: "4+"})
    return cells


def summary_table(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (xc, fc), g in cells.groupby(["xenium_call", "fish_call"]):
        counts = g["egfr_status"].value_counts().reindex(EGFR_ORDER, fill_value=0)
        rows.append({
            "xenium_call": xc, "fish_call": fc, "agreement": g["agreement"].iloc[0], "cells": len(g),
            **{f"egfr_{k}": int(counts[k]) for k in EGFR_ORDER},
            "egfr_amplified_pct": 100 * counts["amplified"] / len(g),
            "egfr_abnormal_pct": 100 * (counts["amplified"] + counts["gain"]) / len(g),
        })
    return pd.DataFrame(rows)


def per_type_table(cells: pd.DataFrame, order: list[str], normal_cep7: int) -> pd.DataFrame:
    rows = []
    for group in order:
        g = cells[cells["group"] == group]
        if g.empty:
            continue
        cep7 = g["cep7_class"].value_counts().reindex(["0", "1", "2", "3", "4+"], fill_value=0)
        egfr = g["egfr_status"].value_counts().reindex(EGFR_ORDER, fill_value=0)
        rows.append({
            "group": group, "xenium_call": g["xenium_call"].iloc[0], "cells": len(g),
            **{f"cep7_{k}": int(v) for k, v in cep7.items()},
            f"cep7_eq_{normal_cep7}_pct": 100 * (g["cep7_spots_nuc"] == normal_cep7).mean(),
            **{f"egfr_{k}": int(egfr[k]) for k in EGFR_ORDER},
            "egfr_amplified_pct": 100 * egfr["amplified"] / len(g),
            "mean_EGFR_transcripts": g["EGFR_transcripts"].mean() if "EGFR_transcripts" in g else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def cep7_figure(per_type: pd.DataFrame, normal_cep7: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    labels = [f"{g}\n(n={n:,})" for g, n in zip(per_type.group, per_type.cells)]
    left = np.zeros(len(per_type))
    for col, color in zip(["0", "1", "2", "3", "4+"], CEP7_COLORS):
        share = per_type[f"cep7_{col}"] / per_type["cells"]
        ax.barh(range(len(per_type)), share, left=left, color=color, label=f"{col} CEP7 spots")
        left += share.to_numpy()
    ax.set_yticks(range(len(per_type)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of cells")
    # Brackets left of the tick labels grouping non-malignant and malignant types.
    trans = ax.get_yaxis_transform()
    for call in ("non-malignant", "malignant"):
        rows = np.flatnonzero(per_type.xenium_call.to_numpy() == call)
        if not len(rows):
            continue
        top, bottom = rows.min() - 0.35, rows.max() + 0.35
        x = -0.17
        ax.plot([x + 0.012, x, x, x + 0.012], [top, top, bottom, bottom], color=INK, lw=1.2,
                transform=trans, clip_on=False)
        ax.text(x - 0.01, (top + bottom) / 2, call, transform=trans, ha="right", va="center", fontsize=11)
    ax.set_title("FISH CEP7 spots per nucleus by Xenium cell type")
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return fig


def agreement_figure(summary: pd.DataFrame, normal_cep7: int) -> plt.Figure:
    order = [("non-malignant", f"CEP7 = {normal_cep7}"), ("non-malignant", f"CEP7 != {normal_cep7}"),
             ("malignant", f"CEP7 != {normal_cep7}"), ("malignant", f"CEP7 = {normal_cep7}")]
    table = summary.set_index(["xenium_call", "fish_call"]).reindex(order).fillna(0)
    labels = [f"Xenium {x}\nFISH {f}\n({'agree' if (x == 'non-malignant') == f.startswith('CEP7 =') else 'disagree'})"
              for x, f in order]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), constrained_layout=True)

    ax = axes[0]
    agree_color, disagree_color = "#2e7d5b", "#9fb3c2"
    bars = ax.bar(range(4), table["cells"], color=[agree_color, disagree_color, agree_color, disagree_color])
    for b, v in zip(bars, table["cells"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{int(v):,}", ha="center", va="bottom")
    ax.set_xticks(range(4), labels, fontsize=9)
    ax.set_ylabel("cells")
    ax.set_yscale("log")
    ax.set_title("Xenium cell type vs FISH chromosome 7 call")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=agree_color), plt.Rectangle((0, 0), 1, 1, color=disagree_color)],
              labels=["agree", "disagree"], frameon=False, loc="upper left")

    ax = axes[1]
    bottom = np.zeros(4)
    totals = table["cells"].replace(0, np.nan).to_numpy()
    for k in EGFR_ORDER:
        share = 100 * table[f"egfr_{k}"].to_numpy() / totals
        ax.bar(range(4), share, bottom=bottom, color=EGFR_COLORS[k], label=f"EGFR {k}")
        bottom += np.nan_to_num(share)
    for i, (amp, n) in enumerate(zip(table["egfr_amplified"], table["cells"])):
        if n:
            ax.text(i, 101, f"{int(amp):,} amplified\n({100 * amp / n:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(4), labels, fontsize=9)
    ax.set_ylim(0, 118)
    ax.set_ylabel("share of cells (%)")
    ax.set_title("EGFR FISH status in each group")
    ax.legend(frameon=False, loc="lower right")
    return fig


def egfr_by_type_figure(cells: pd.DataFrame, order: list[str], normal_cep7: int) -> plt.Figure:
    nm = cells[cells.xenium_call == "non-malignant"]
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    groups = [g for g in order if g in set(nm.group)]
    width = 0.38
    for j, (agree, hatch) in enumerate((("agree", None), ("disagree", "///"))):
        pct, n_amp, n = [], [], []
        for g in groups:
            sub = nm[(nm.group == g) & (nm.agreement == agree)]
            n.append(len(sub))
            n_amp.append(int((sub.egfr_status == "amplified").sum()))
            pct.append(100 * n_amp[-1] / len(sub) if len(sub) else 0)
        x = np.arange(len(groups)) + (j - 0.5) * width
        ax.bar(x, pct, width, color="#c0392b" if agree == "agree" else "white", edgecolor="#c0392b", hatch=hatch,
               label=f"FISH CEP7 {'=' if agree == 'agree' else '!='} {normal_cep7} ({agree})")
        for xi, p, a, t in zip(x, pct, n_amp, n):
            ax.text(xi, p, f"{a}/{t}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylabel("EGFR amplified (% of cells)")
    ax.set_title("EGFR amplification in Xenium non-malignant cells, by type and FISH agreement (labels: amplified / cells)")
    ax.legend(frameon=False)
    return fig


# --------------------------------------------------------------------------
# How well do red / green spots separate malignant from non-malignant?
# --------------------------------------------------------------------------

SPOT_MAX = 8
MAL_COLOR, NON_COLOR = "#c0392b", "#2e7d5b"  # distribution curves: malignant vs non-malignant
EGFR_RED, CEP7_GREEN, RATIO_COLOR = "#c0392b", "#2e7d5b", "#5b4b8a"  # ROC curves: one colour per FISH measure


def _features(cells: pd.DataFrame) -> dict[str, pd.Series]:
    e, c = cells["egfr_spots_nuc"], cells["cep7_spots_nuc"]
    return {
        "EGFR spots": e,
        "CEP7 spots": c,
        "EGFR/CEP7 spots": e / c.where(c > 0),  # cells with at least one CEP7 spot
        "EGFR/CEP7 intensity": cells["egfr_cep7_nuc_ratio"],
    }


def _auc(score: np.ndarray, is_mal: np.ndarray) -> float:
    """Probability that a random malignant cell scores higher than a random non-malignant one (ties count half)."""
    r = rankdata(score)
    n1, n0 = is_mal.sum(), (~is_mal).sum()
    return float((r[is_mal].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _auc_ci(score: np.ndarray, is_mal: np.ndarray, reps: int = 200, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    mal, non = np.flatnonzero(is_mal), np.flatnonzero(~is_mal)
    vals = []
    for _ in range(reps):
        idx = np.r_[rng.choice(mal, len(mal)), rng.choice(non, len(non))]
        vals.append(_auc(score[idx], is_mal[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def separation_stats(cells: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, float]]]:
    is_mal = (cells["xenium_call"] == "malignant").to_numpy()
    rows, roc_inputs = [], {}
    for name, values in _features(cells).items():
        x = values.to_numpy(dtype=float)
        ok = np.isfinite(x)
        x, y = x[ok], is_mal[ok]
        auc = _auc(x, y)
        lo, hi = _auc_ci(x, y)
        p = mannwhitneyu(x[y], x[~y]).pvalue
        row = {"feature": name, "auc": auc, "auc_ci95_low": lo, "auc_ci95_high": hi, "mannwhitney_p": p,
               "median_malignant": float(np.median(x[y])), "median_non_malignant": float(np.median(x[~y])),
               "mean_malignant": float(x[y].mean()), "mean_non_malignant": float(x[~y].mean())}
        if name in ("EGFR spots", "CEP7 spots"):
            table = pd.crosstab(np.clip(x, 0, SPOT_MAX), y)
            row["chi2_p"] = chi2_contingency(table.to_numpy())[1]
        if name in ("EGFR spots", "EGFR/CEP7 spots"):
            best = max(((t, (x[y] >= t).mean(), (x[~y] < t).mean()) for t in np.unique(x)[1:]), key=lambda v: v[1] + v[2])
            row |= {"best_cut_at_or_above": best[0], "sensitivity": best[1], "specificity": best[2]}
        rows.append(row)
        roc_inputs[name] = (x, y)
    return pd.DataFrame(rows), roc_inputs


def _share(values: pd.Series, support: np.ndarray) -> np.ndarray:
    return values.value_counts(normalize=True).reindex(support, fill_value=0).to_numpy()


def distribution_figures(cells: pd.DataFrame, stats: pd.DataFrame, roc_inputs: dict) -> list[tuple[str, plt.Figure]]:
    """Malignant vs non-malignant spot distributions and the ROC curves, one figure each."""
    mal = cells[cells.xenium_call == "malignant"]
    non = cells[cells.xenium_call == "non-malignant"]
    groups = ((mal, MAL_COLOR, "malignant"), (non, NON_COLOR, "non-malignant"))
    auc = stats.set_index("feature")["auc"]
    out = []

    for col, xlabel, feat in (("egfr_spots_nuc", "EGFR spots in the nucleus", "EGFR spots"),
                              ("cep7_spots_nuc", "CEP7 spots in the nucleus", "CEP7 spots")):
        fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
        support = np.arange(SPOT_MAX + 1)
        for sub, color, label in groups:
            ax.plot(support, 100 * _share(sub[col].clip(upper=SPOT_MAX), support), "-o", color=color, lw=2,
                    label=f"Xenium {label} (n={len(sub):,}, mean {sub[col].mean():.2f})")
        ax.set_xticks(support, [str(v) for v in support[:-1]] + [f"{SPOT_MAX}+"])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("share of cells (%)")
        ax.set_title(f"{xlabel}: AUC = {auc[feat]:.3f}")
        ax.legend(frameon=False)
        out.append((f"{feat} distribution malignant vs non-malignant", fig))

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    grid = np.linspace(0, 4, 401)
    bandwidth = 0.15
    for sub, color, label in groups:
        counted = sub[sub["cep7_spots_nuc"] > 0]
        ratio = (counted["egfr_spots_nuc"] / counted["cep7_spots_nuc"]).clip(upper=4).to_numpy()
        kernel = np.exp(-0.5 * ((grid[:, None] - ratio[None, :]) / bandwidth) ** 2)
        density = kernel.sum(1) / (len(ratio) * bandwidth * np.sqrt(2 * np.pi))
        ax.plot(grid, density, color=color, lw=2, label=f"Xenium {label} (n={len(counted):,})")
    ax.axvline(1, color="#7f8c8d", ls="--", lw=1.2)
    ax.text(1.03, ax.get_ylim()[1] * 0.95, "ratio = 1", color="#7f8c8d", va="top")
    ax.set_xlim(0, 4)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], ["0", "0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4+"])
    ax.set_xlabel("EGFR/CEP7 spots (cells with at least one CEP7 spot)")
    ax.set_ylabel("density of cells")
    ax.set_title(f"EGFR/CEP7: AUC = {auc['EGFR/CEP7 spots']:.3f}")
    ax.legend(frameon=False)
    out.append(("EGFR CEP7 spot ratio distribution malignant vs non-malignant", fig))

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    roc_colors = {"EGFR spots": EGFR_RED, "CEP7 spots": CEP7_GREEN, "EGFR/CEP7 spots": RATIO_COLOR,
                  "EGFR/CEP7 intensity": "#e0a33a"}
    for name, (x, y) in roc_inputs.items():
        cuts = np.unique(x)[::-1]
        tpr = [(x[y] >= t).mean() for t in cuts]
        fpr = [(x[~y] >= t).mean() for t in cuts]
        ax.plot(np.r_[0, fpr, 1], np.r_[0, tpr, 1], color=roc_colors.get(name, "#7f8c8d"), lw=1.8,
                label=f"{name}: AUC {auc[name]:.3f}")
    ax.plot([0, 1], [0, 1], color="#c9ced2", ls=":", lw=1)
    ax.set_xlabel("FP rate: Xenium non-malignant cells called malignant by FISH")
    ax.set_ylabel("TP rate: Xenium malignant cells called malignant by FISH")
    ax.set_title("ROC: calling a cell malignant from FISH")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    out.append(("ROC malignant from FISH", fig))
    return out


def _uri(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def save_plots(out_dir: Path, prefix: str, figures: list[tuple[str, plt.Figure]]) -> None:
    """Save every figure as its own PNG under <out_dir>/plots."""
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for title, fig in figures:
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        fig.savefig(plot_dir / f"{prefix}_{slug}.png", dpi=130)


def write_report(out_dir: Path, notes: list[str], tables: list[tuple[str, pd.DataFrame]], figures: list[tuple[str, plt.Figure]]) -> None:
    fmt = lambda v: f"{v:.1f}"  # noqa: E731
    body = "<ul>" + "".join(f"<li>{n}</li>" for n in notes) + "</ul>"
    body += "".join(f"<h2>{t}</h2><div class='scroll'>{df.to_html(index=False, float_format=fmt, border=0)}</div>" for t, df in tables)
    body += "".join(f"<h2>{t}</h2><img src='{_uri(f)}' alt='{t}'>" for t, f in figures)
    (out_dir / "nonmalignant_report.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Non-malignant cells vs FISH</title>
<style>
 body {{ font: 15px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #1c2328; background: #f6f7f8; margin: 0; }}
 main {{ max-width: 1150px; margin: 0 auto; padding: 24px 16px 48px; }}
 h1 {{ font-size: 24px; }} h2 {{ font-size: 18px; margin-top: 32px; }}
 table {{ border-collapse: collapse; background: #fff; font-size: 13px; }}
 th, td {{ border: 1px solid #dde1e4; padding: 4px 8px; text-align: right; }} th {{ background: #eef1f3; }}
 td:first-child, th:first-child {{ text-align: left; }}
 img {{ max-width: 100%; height: auto; background: #fff; border: 1px solid #dde1e4; }}
 .scroll {{ overflow-x: auto; }}
</style></head><body><main><h1>Xenium non-malignant cells vs FISH</h1>{body}</main></body></html>""", encoding="utf-8")
    with PdfPages(out_dir / "nonmalignant_report.pdf") as pdf:
        page = plt.figure(figsize=(11, 8.5))
        text = ["Xenium non-malignant cells vs FISH", ""] + [textwrap.fill(f"- {n}", 150, subsequent_indent="  ") for n in notes]
        for t, df in tables:
            text += ["", t, df.to_string(index=False, float_format=fmt)]
        page.text(0.03, 0.97, "\n".join(text), va="top", family="monospace", fontsize=6)
        pdf.savefig(page)
        plt.close(page)
        for _, fig in figures:
            pdf.savefig(fig)


def main() -> None:
    args = _parse_args()
    out_dir = args.output_dir or args.cells.parent
    cells = load(args)
    order = [g for g in args.non_malignant + args.malignant if g in set(cells.group)]
    summary = summary_table(cells)
    per_type = per_type_table(cells, order, args.normal_cep7)

    nm = cells[cells.xenium_call == "non-malignant"]
    agree = nm[nm.agreement == "agree"]
    disagree = nm[nm.agreement == "disagree"]
    reasons = disagree["cep7_class"].value_counts().reindex(["0", "1", "3", "4+"], fill_value=0)
    mal_normal = (cells[cells.xenium_call == "malignant"]["cep7_spots_nuc"] == args.normal_cep7).mean()
    headline = [
        f"Xenium non-malignant cells: {len(nm):,}. FISH shows exactly {args.normal_cep7} CEP7 spots in {len(agree):,} "
        f"({len(agree) / max(len(nm), 1):.0%}, agree) and another count in {len(disagree):,} (disagree).",
        "Disagreements by CEP7 count: " + ", ".join(f"{k} spots {v:,}" for k, v in reasons.items()) + ".",
        f"For comparison, {mal_normal:.0%} of Xenium malignant cells also show exactly {args.normal_cep7} CEP7 spots, "
        "so a CEP7 count of 2 on its own separates the two poorly.",
        f"EGFR amplified among non-malignant cells: {int((agree.egfr_status == 'amplified').sum()):,} of {len(agree):,} that agree "
        f"({(agree.egfr_status == 'amplified').mean():.1%}), {int((disagree.egfr_status == 'amplified').sum()):,} of "
        f"{len(disagree):,} that disagree ({(disagree.egfr_status == 'amplified').mean():.1%}).",
    ]
    notes = headline + [
        f"Malignant groups: {', '.join(args.malignant)}. Non-malignant groups: {', '.join(args.non_malignant)}.",
        f"FISH normal: exactly {args.normal_cep7} CEP7 spots in the Xenium nucleus outline. Agree means Xenium non-malignant with "
        f"FISH normal, or Xenium malignant with FISH not normal.",
        f"EGFR amplified: EGFR/CEP7 spots of at least {args.amp_ratio:g} with at least {args.amp_min_egfr} EGFR spots, or at least "
        f"{args.amp_min_egfr_no_cep7} EGFR spots with no CEP7 spot. EGFR gain: not amplified but at least {args.gain_egfr} EGFR spots.",
        "A 5 um section cuts through most nuclei, so normal cells often show fewer than 2 CEP7 spots, and at 0.49 um per pixel "
        "nearby spots can merge. Both push normal cells into the 'disagree' group; the spot counts are approximate.",
        "Cells used: present in FISH, not saturated, not on debris, and listed in the cell-type table.",
    ]
    for n in headline:
        print(n)

    figures = [
        ("CEP7 spots per nucleus by cell type", cep7_figure(per_type, args.normal_cep7)),
        ("Agreement and EGFR status", agreement_figure(summary, args.normal_cep7)),
        ("EGFR amplification in non-malignant cells", egfr_by_type_figure(cells, order, args.normal_cep7)),
    ]
    stats, roc_inputs = separation_stats(cells)
    figures += distribution_figures(cells, stats, roc_inputs)
    best = stats.loc[stats["auc"].idxmax()]
    notes += [
        "Separation: AUC is the chance that a random Xenium-malignant cell scores higher than a random non-malignant one "
        "(0.5 = no separation, 1 = perfect). With tens of thousands of cells even tiny differences have very small p-values, "
        "so judge separation by AUC, not p.",
        f"Best single FISH score: {best.feature}, AUC {best.auc:.3f} (95% CI {best.auc_ci95_low:.3f}-{best.auc_ci95_high:.3f}).",
    ]
    print(stats[["feature", "auc", "auc_ci95_low", "auc_ci95_high", "mannwhitney_p", "mean_malignant", "mean_non_malignant"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    stats.to_csv(out_dir / "nonmalignant_separation_stats.csv", index=False, float_format="%.4g")
    tables = [("Agreement groups", summary), ("Per cell type", per_type), ("Separating malignant from non-malignant", stats)]
    cols = ["cell_id", "group", "xenium_call", "fish_call", "agreement", "cep7_spots_nuc", "egfr_spots_nuc",
            "egfr_status", "egfr_cep7_nuc_ratio", "EGFR_transcripts", "fish_x", "fish_y", "x_centroid", "y_centroid"]
    cells[[c for c in cols if c in cells]].to_csv(out_dir / "nonmalignant_cells.csv", index=False, float_format="%.4f")
    pd.concat([summary.assign(table="agreement"), per_type.assign(table="per_type")]).to_csv(
        out_dir / "nonmalignant_summary.csv", index=False, float_format="%.3f")
    save_plots(out_dir, "nonmalignant", figures)
    write_report(out_dir, notes, tables, figures)
    print(f"outputs in {out_dir}: nonmalignant_cells.csv, nonmalignant_summary.csv, nonmalignant_report.html, .pdf")


if __name__ == "__main__":
    main()
