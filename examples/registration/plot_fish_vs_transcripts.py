"""Plot Xenium transcription against FISH amplification, per cell.

Reads ``cells.csv`` from ``fish_to_xenium_cells.py`` and writes
``transcription_vs_amplification.png`` / ``.pdf`` and the binned table as CSV.

Amplification is measured three ways: the probe / control nuclear intensity
ratio (EGFR / CEP7), the number of EGFR spots in the nucleus, and the
clinical-style spot-count ratio (EGFR spots / CEP7 spots, cut-offs 1 and 2).
Transcription is shown as raw transcripts per cell and as the share of the
cell's transcripts, which removes the effect of cell size and capture.

Run:
    uv run --no-sync python examples/registration/plot_fish_vs_transcripts.py \
        --cells outputs/.../refine/cells/cells.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

INK = "#1c2328"
LINE = "#c0392b"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", type=Path, required=True, help="cells.csv from fish_to_xenium_cells.py")
    p.add_argument("--gene", default="EGFR", help="Gene whose <gene>_transcripts column is plotted.")
    p.add_argument("--bins", type=int, default=10, help="Number of equal-sized amplification bins.")
    p.add_argument("--max-spots", type=int, default=6, help="Spot counts at or above this are pooled.")
    p.add_argument("--heatmap-max", type=int, default=8, help="Spot counts at or above this are pooled in the heatmap.")
    p.add_argument("--min-cells", type=int, default=30, help="Heatmap squares with fewer cells are left blank.")
    p.add_argument("--output-dir", type=Path, default=None, help="Default: folder of cells.csv")
    return p.parse_args()


def _mean_ci(values: pd.Series) -> tuple[float, float]:
    return float(values.mean()), float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def _binned(cells: pd.DataFrame, key: pd.Series, x: str, tx: str) -> pd.DataFrame:
    rows = []
    for name, g in cells.groupby(key, observed=True):
        tm, tc = _mean_ci(g[tx])
        sm, sc = _mean_ci(g["share"])
        rows.append({"group": name, "x": g[x].median(), "cells": len(g),
                     "transcripts_mean": tm, "transcripts_ci95": tc, "share_mean": sm, "share_ci95": sc})
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    out_dir = args.output_dir or args.cells.parent
    tx = f"{args.gene}_transcripts"
    cells = pd.read_csv(args.cells)
    keep = (cells["nucleus_area_fish_px"] > 0) & cells["in_fish"] & ~cells["saturated"]
    if "debris" in cells:
        keep &= ~cells["debris"]
    cells = cells[keep]
    cells = cells[np.isfinite(cells["egfr_cep7_nuc_ratio"])].copy()
    cells["share"] = 100.0 * cells[tx] / cells["transcript_counts"].clip(lower=1)
    n = len(cells)

    ratio = cells["egfr_cep7_nuc_ratio"]
    by_ratio = _binned(cells, pd.qcut(ratio, args.bins, labels=False), "egfr_cep7_nuc_ratio", tx)
    spots = cells["egfr_spots_nuc"].clip(upper=args.max_spots)
    by_spots = _binned(cells, spots, "egfr_spots_nuc", tx)
    rho_ratio = spearmanr(ratio, cells["share"]).statistic
    rho_spots = spearmanr(cells["egfr_spots_nuc"], cells["share"]).statistic

    # Clinical-style spot-count ratio; undefined for nuclei without a CEP7 spot.
    counted = cells[cells["cep7_spots_nuc"] > 0].copy()
    counted["spot_ratio"] = counted["egfr_spots_nuc"] / counted["cep7_spots_nuc"]
    edges = [0, 0.5, 1, 1.5, 2, 3, np.inf]
    names = ["<0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", ">=3"]
    counted["ratio_class"] = pd.cut(counted["spot_ratio"], edges, right=False, labels=names)
    by_class = _binned(counted, counted["ratio_class"], "spot_ratio", tx)
    by_class["x"] = np.arange(len(by_class))
    rho_count = spearmanr(counted["spot_ratio"], counted["share"]).statistic

    fig, grid = plt.subplots(2, 3, figsize=(19, 11.6), constrained_layout=True)
    axes = grid[0]

    ax = axes[0]
    lo, hi = ratio.quantile([0.005, 0.995])
    top = cells["share"].quantile(0.995)
    keep = ratio.between(lo, hi) & (cells["share"] <= top)
    hb = ax.hexbin(ratio[keep], cells["share"][keep], gridsize=55, bins="log", cmap="Greys", mincnt=1)
    ax.plot(by_ratio["x"], by_ratio["share_mean"], "-o", color=LINE, lw=2, ms=5, label=f"mean per {100 // args.bins}% bin")
    ax.set_xlabel("EGFR / CEP7 nuclear FISH intensity ratio")
    ax.set_ylabel(f"{args.gene} share of the cell's transcripts (%)")
    ax.set_title(f"Every cell (n = {n:,}), Spearman rho = {rho_ratio:.2f}")
    ax.legend(frameon=False, loc="upper left")
    fig.colorbar(hb, ax=ax, label="cells per hexagon", shrink=0.8)

    for ax, table, xlabel, title, rho in (
        (axes[1], by_ratio, "EGFR / CEP7 nuclear FISH intensity ratio (bin median)",
         f"Cells grouped into {args.bins} equal bins by FISH ratio", rho_ratio),
        (axes[2], by_spots, "EGFR FISH spots in the nucleus",
         "Cells grouped by nuclear EGFR spot count", rho_spots),
    ):
        ax.errorbar(table["x"], table["transcripts_mean"], yerr=table["transcripts_ci95"], fmt="-o",
                    color=INK, lw=2, ms=6, capsize=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"mean {args.gene} transcripts per cell", color=INK)
        ax.set_ylim(bottom=0)
        twin = ax.twinx()
        twin.errorbar(table["x"], table["share_mean"], yerr=table["share_ci95"], fmt="--s",
                      color=LINE, lw=1.6, ms=5, capsize=3)
        twin.set_ylabel(f"mean {args.gene} share of transcripts (%)", color=LINE)
        twin.tick_params(axis="y", colors=LINE)
        twin.set_ylim(bottom=0)
        ax.set_title(f"{title} (rho = {rho:.2f})")
    if args.max_spots in set(by_spots["x"].round().astype(int)):
        ticks = list(range(args.max_spots + 1))
        axes[2].set_xticks(ticks, [str(t) for t in ticks[:-1]] + [f"{args.max_spots}+"])

    ax = grid[1, 0]
    # Bins of width 0.5, left edge included (a ratio of exactly 1 falls in 1-1.5); everything >= 5 pooled.
    width, last = 0.5, 5.0
    edges = np.arange(0, last + width, width)
    heights = np.histogram(counted["spot_ratio"].clip(upper=last - 1e-9), bins=edges)[0]
    pooled = int((counted["spot_ratio"] >= last).sum())
    ax.bar(edges[:-1], heights, width=width, align="edge", color="#7f8c8d", edgecolor="white")
    ax.bar(last, pooled, width=width, align="edge", color="#4d5656", edgecolor="white")
    for cut, label, height in ((1, "1", 0.95), (2, "2 (amplified)", 0.85)):
        ax.axvline(cut, color=LINE, lw=1.5, ls="--")
        ax.text(cut + 0.05, ax.get_ylim()[1] * height, f"{(counted['spot_ratio'] >= cut).mean():.0%} at or above {label}",
                color=LINE, va="top", backgroundcolor="white")
    ax.set_xticks(list(np.arange(0, last + 0.01, 1.0)) + [last + width / 2], [f"{t:g}" for t in np.arange(0, last + 0.01, 1.0)] + ["5+"])
    ax.set_xlim(0, last + width)
    ax.set_xlabel("EGFR spots / CEP7 spots in the nucleus (bins of 0.5; darkest bar: 5 and above)")
    ax.set_ylabel("cells")
    ax.set_title(f"Spot-count ratio, cells with at least one CEP7 spot (n = {len(counted):,})")

    ax = grid[1, 1]
    pos = np.arange(len(by_class))
    ax.bar(pos - 0.2, by_class["transcripts_mean"], width=0.4, color=INK, yerr=by_class["transcripts_ci95"],
           capsize=3, error_kw={"ecolor": "#7f8c8d"}, label=f"{args.gene} transcripts per cell")
    for x, (v, c) in enumerate(zip(by_class["transcripts_mean"], by_class["cells"])):
        ax.text(x - 0.2, v * 0.5, f"n={c:,}", ha="center", va="center", color="white", fontsize=8, rotation=90)
    ax.set_xticks(pos, by_class["group"].astype(str))
    ax.set_xlabel("EGFR / CEP7 spot-count ratio")
    ax.set_ylabel(f"mean {args.gene} transcripts per cell", color=INK)
    ax.set_ylim(bottom=0)
    twin = ax.twinx()
    twin.bar(pos + 0.2, by_class["share_mean"], width=0.4, fill=False, hatch="///", edgecolor=LINE, linewidth=1.2,
             yerr=by_class["share_ci95"], capsize=3, error_kw={"ecolor": LINE}, label=f"{args.gene} share of transcripts")
    twin.set_ylabel(f"mean {args.gene} share of transcripts (%)", color=LINE)
    twin.tick_params(axis="y", colors=LINE)
    twin.set_ylim(bottom=0)
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    ax.legend(handles, [h.get_label() for h in handles], frameon=False, loc="upper left")
    ax.set_title(f"Cells grouped by spot-count ratio (rho = {rho_count:.2f})")

    ax = grid[1, 2]
    top = args.heatmap_max
    # All usable cells, including those without a CEP7 spot (bottom row).
    e = cells["egfr_spots_nuc"].clip(upper=top)
    c = cells["cep7_spots_nuc"].clip(upper=top)
    mean_tx = cells.groupby([c, e])[tx].mean().unstack().reindex(index=range(top + 1), columns=range(top + 1))
    n_cells = cells.groupby([c, e]).size().unstack().reindex(index=range(top + 1), columns=range(top + 1))
    mean_tx = mean_tx.where(n_cells >= args.min_cells)
    values = mean_tx.to_numpy(dtype=float)
    im = ax.imshow(values, origin="lower", cmap="viridis", aspect="auto", extent=(-0.5, top + 0.5, -0.5, top + 0.5))
    for (ci, ei), v in np.ndenumerate(values):
        if np.isfinite(v):
            ax.text(ei, ci, f"{v:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if v < np.nanmax(values) * 0.6 else INK)
    xs = np.linspace(-0.5, top + 0.5, 50)
    ax.plot(xs, xs, color="white", lw=1, ls=":")
    ax.plot(xs, xs / 2, color=LINE, lw=1.2, ls="--")
    ax.set_xlim(-0.5, top + 0.5)
    ax.set_ylim(-0.5, top + 0.5)
    labels = [str(t) for t in range(top)] + [f"{top}+"]
    ax.set_xticks(range(top + 1), labels)
    ax.set_yticks(range(top + 1), labels)
    ax.set_xlabel("EGFR spots in the nucleus")
    ax.set_ylabel("CEP7 spots in the nucleus (0 = no CEP7 spot)")
    ax.set_title(f"Mean {args.gene} transcripts (Xenium) by FISH spot counts\n"
                 f"blank: fewer than {args.min_cells} cells; dotted: ratio 1, dashed: ratio 2")
    fig.colorbar(im, ax=ax, label=f"mean {args.gene} transcripts per cell (Xenium)", shrink=0.8)
    fig.suptitle(
        f"{args.gene} transcription (Xenium) vs EGFR amplification (FISH), per cell. "
        "Black: transcripts per cell. Red dashed / hatched: share of transcripts. Error bars: 95% CI of the mean."
    )
    fig.savefig(out_dir / "transcription_vs_amplification.png", dpi=130)
    fig.savefig(out_dir / "transcription_vs_amplification.pdf")

    by_ratio.insert(0, "grouping", "ratio_bin")
    by_spots.insert(0, "grouping", "nuclear_spots")
    by_class.insert(0, "grouping", "spot_count_ratio")
    pd.concat([by_ratio, by_spots, by_class]).to_csv(out_dir / "transcription_vs_amplification.csv", index=False, float_format="%.4f")
    print(f"cells used: {n:,}; Spearman rho (ratio vs share) = {rho_ratio:.3f}, (spots vs share) = {rho_spots:.3f}")
    print(f"spot-count ratio: {len(counted):,} cells with a CEP7 spot, {(counted['spot_ratio'] >= 1).mean():.0%} >= 1, "
          f"{(counted['spot_ratio'] >= 2).mean():.0%} >= 2, rho vs share = {rho_count:.3f}")
    print(f"outputs in {out_dir}: transcription_vs_amplification.png, .pdf, .csv")


if __name__ == "__main__":
    main()
