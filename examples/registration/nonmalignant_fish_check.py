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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

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
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
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
    nm = per_type[per_type.xenium_call == "non-malignant"]
    if len(nm):
        ax.axhline(len(nm) - 0.5, color=INK, lw=1)
    ax.set_title(f"CEP7 (green) spots per nucleus by Xenium cell type; dark green = {normal_cep7} spots "
                 "(normal). Non-malignant types above the line.")
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
    bars = ax.bar(range(4), table["cells"], color=["#2e7d5b", "#9fb3c2", "#2e7d5b", "#9fb3c2"])
    for b, v in zip(bars, table["cells"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{int(v):,}", ha="center", va="bottom")
    ax.set_xticks(range(4), labels, fontsize=9)
    ax.set_ylabel("cells")
    ax.set_yscale("log")
    ax.set_title("Xenium cell type vs FISH chromosome 7 call (green: agree, grey: disagree)")

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


def _uri(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


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
    tables = [("Agreement groups", summary), ("Per cell type", per_type)]
    cols = ["cell_id", "group", "xenium_call", "fish_call", "agreement", "cep7_spots_nuc", "egfr_spots_nuc",
            "egfr_status", "egfr_cep7_nuc_ratio", "EGFR_transcripts", "fish_x", "fish_y", "x_centroid", "y_centroid"]
    cells[[c for c in cols if c in cells]].to_csv(out_dir / "nonmalignant_cells.csv", index=False, float_format="%.4f")
    pd.concat([summary.assign(table="agreement"), per_type.assign(table="per_type")]).to_csv(
        out_dir / "nonmalignant_summary.csv", index=False, float_format="%.3f")
    write_report(out_dir, notes, tables, figures)
    print(f"outputs in {out_dir}: nonmalignant_cells.csv, nonmalignant_summary.csv, nonmalignant_report.html, .pdf")


if __name__ == "__main__":
    main()
