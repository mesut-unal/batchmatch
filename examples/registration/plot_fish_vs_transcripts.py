"""Plot Xenium transcription against FISH amplification, per cell.

Reads ``cells.csv`` from ``fish_to_xenium_cells.py`` and writes
``transcription_vs_amplification.png`` / ``.pdf`` and the binned table as CSV.

Amplification is measured two ways: the probe / control nuclear intensity
ratio (EGFR / CEP7), and the number of probe spots in the nucleus.
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
    cells = cells[(cells["nucleus_area_fish_px"] > 0) & cells["in_fish"] & ~cells["saturated"]]
    cells = cells[np.isfinite(cells["egfr_cep7_nuc_ratio"])].copy()
    cells["share"] = 100.0 * cells[tx] / cells["transcript_counts"].clip(lower=1)
    n = len(cells)

    ratio = cells["egfr_cep7_nuc_ratio"]
    by_ratio = _binned(cells, pd.qcut(ratio, args.bins, labels=False), "egfr_cep7_nuc_ratio", tx)
    spots = cells["egfr_spots_nuc"].clip(upper=args.max_spots)
    by_spots = _binned(cells, spots, "egfr_spots_nuc", tx)
    rho_ratio = spearmanr(ratio, cells["share"]).statistic
    rho_spots = spearmanr(cells["egfr_spots_nuc"], cells["share"]).statistic

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), constrained_layout=True)

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

    fig.suptitle(
        f"{args.gene} transcription (Xenium) vs EGFR amplification (FISH), per cell. "
        "Solid black: transcripts per cell. Dashed red: share of transcripts. Bars: 95% CI of the mean."
    )
    fig.savefig(out_dir / "transcription_vs_amplification.png", dpi=130)
    fig.savefig(out_dir / "transcription_vs_amplification.pdf")

    by_ratio.insert(0, "grouping", "ratio_bin")
    by_spots.insert(0, "grouping", "nuclear_spots")
    pd.concat([by_ratio, by_spots]).to_csv(out_dir / "transcription_vs_amplification.csv", index=False, float_format="%.4f")
    print(f"cells used: {n:,}; Spearman rho (ratio vs share) = {rho_ratio:.3f}, (spots vs share) = {rho_spots:.3f}")
    print(f"outputs in {out_dir}: transcription_vs_amplification.png, .pdf, .csv")


if __name__ == "__main__":
    main()
