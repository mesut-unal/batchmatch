"""Smoothed spatial maps of EGFR FISH signal vs Xenium EGFR transcripts.

The per-cell scatter map from ``fish_to_xenium_cells.py`` is too noisy to
compare by eye. This script bins usable cells (in FISH, not saturated, not
debris) on a square grid in FISH pixels, smooths every map with the same
Gaussian, and turns each into a per-cell average so dense regions don't
look "high" just because they hold more cells:

- EGFR FISH spots per nucleus
- EGFR/CEP7 spots (sum of EGFR spots / sum of CEP7 spots in the neighbourhood)
- EGFR FISH nuclear mean intensity
- Xenium EGFR transcripts per cell

It reports the Spearman correlation between the FISH maps and the transcript
map across grid bins that hold at least ``--min-cells`` cells, and draws a
red (FISH) / green (transcripts) overlay where yellow means both are high.

Outputs under ``<cells dir>/plots``: ``cells_spatial_density_*.png`` and
``spatial_density_stats.csv`` next to ``cells.csv``.

Which spatial plot is which (all under ``cells/plots``):

    file prefix                 made by                         unit                                   outline from
    cells_spatial_pattern_*     fish_to_xenium_cells.py         one dot per cell, raw values            Xenium cell centroids
    cells_spatial_density_*     spatial_density.py              per-cell averages (FISH spots per       Xenium cells usable in FISH
                                                                nucleus, EGFR/CEP7 spots, nuclear
                                                                intensity, transcripts per cell)
    cells_intensity_density_*   spatial_intensity_density.py    per tissue area (raw EGFR FISH pixel    FISH DAPI tissue mask
                                                                signal, transcripts per 100 um^2)

In both density families, ``_agreement_`` files compare one FISH map with that
family's transcript map (same bins, same smoothing), and ``_malignant_`` /
``_non_malignant_`` restrict it to those Xenium cell types.

Run:
    uv run --no-sync python examples/registration/spatial_density.py \
        --cells outputs/.../refine/cells/cells.csv \
        --celltypes /path/to/annotations/p46p_celltype.csv   # optional: malignant / non-malignant agreement maps
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import spearmanr

FISH_UM_PER_PX = 0.4875
MALIGNANT = ["AC-like", "MES-like", "OPC-like", "Cycling"]
NON_MALIGNANT = ["Endothelial", "Pericyte", "Myeloid", "Neuron", "Oligodendrocyte", "T cell"]
FISH_MAPS = ("EGFR FISH spots per nucleus", "EGFR/CEP7 FISH spots")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", type=Path, required=True, help="cells.csv from fish_to_xenium_cells.py")
    p.add_argument("--gene", default="EGFR", help="Xenium gene column <gene>_transcripts")
    p.add_argument("--bin-px", type=float, default=50, help="grid bin size in FISH px (default 50 = ~24 um)")
    p.add_argument("--sigma-bins", type=float, default=2.0, help="Gaussian smoothing in bins (default 2 = ~50 um)")
    p.add_argument("--min-cells", type=float, default=3, help="smoothed cells per bin needed to show / correlate a bin")
    p.add_argument("--celltypes", type=Path, default=None,
                   help="CSV with columns cell_id,group; adds agreement maps for malignant and non-malignant cells")
    p.add_argument("--malignant", nargs="+", default=MALIGNANT, help="groups counted as malignant")
    p.add_argument("--non-malignant", nargs="+", default=NON_MALIGNANT, help="groups counted as non-malignant")
    p.add_argument("--min-cells-subset", type=float, default=1,
                   help="smoothed cells per bin needed for the malignant / non-malignant maps (sparser, default 1)")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def smoothed_grid(cells: pd.DataFrame, values: pd.Series | None, shape: tuple[int, int], bin_px: float, sigma: float) -> np.ndarray:
    iy = np.clip((cells.fish_y.to_numpy() // bin_px).astype(int), 0, shape[0] - 1)
    ix = np.clip((cells.fish_x.to_numpy() // bin_px).astype(int), 0, shape[1] - 1)
    w = np.ones(len(cells)) if values is None else values.to_numpy(dtype=float)
    grid = np.zeros(shape)
    np.add.at(grid, (iy, ix), np.nan_to_num(w))
    return ndi.gaussian_filter(grid, sigma)


def build_maps(cells: pd.DataFrame, shape: tuple[int, int], args: argparse.Namespace, min_cells: float) -> tuple[dict, np.ndarray]:
    """Smoothed per-cell averages on the grid; bins with fewer than ``min_cells`` smoothed cells are NaN."""
    grid = lambda col: smoothed_grid(cells, None if col is None else cells[col], shape, args.bin_px, args.sigma_bins)
    n = grid(None)
    show = n >= min_cells
    with np.errstate(invalid="ignore", divide="ignore"):
        maps = {
            "EGFR FISH spots per nucleus": grid("egfr_spots_nuc") / n,
            "EGFR/CEP7 FISH spots": grid("egfr_spots_nuc") / grid("cep7_spots_nuc"),
            "EGFR FISH nuclear mean intensity": grid("egfr_nuc_mean") / n,
            f"{args.gene} transcripts per cell (Xenium)": grid(f"{args.gene}_transcripts") / n,
        }
    for m in maps.values():
        m[~show] = np.nan
    return maps, show


def agreement_figure(fish: np.ndarray, tx: np.ndarray, extent, title: str, um: float) -> tuple[plt.Figure, dict]:
    """Percentile-rank both maps across bins; agreement = 1 - |rank difference| (1 = same level, 0 = opposite ends)."""
    ok = np.isfinite(fish) & np.isfinite(tx)
    agree = np.full(fish.shape, np.nan)
    pf = pd.Series(fish[ok]).rank(pct=True).to_numpy()
    pt = pd.Series(tx[ok]).rank(pct=True).to_numpy()
    agree[ok] = 1 - np.abs(pf - pt)
    r = spearmanr(fish[ok], tx[ok]).statistic
    med = float(np.median(agree[ok]))
    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    ax.set_facecolor("#e8eaec")
    im = ax.imshow(agree, extent=extent, cmap="Blues", vmin=0, vmax=1, interpolation="bilinear")
    ax.set_title(f"{title}\nmedian agreement = {med:.2f}, spatial Spearman rho = {r:.2f}  (smoothed ~{um:.0f} µm)")
    ax.set_xlabel("FISH x (px)")
    ax.set_ylabel("FISH y (px)")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("agreement = 1 − |FISH percentile − transcript percentile|")
    cb.set_ticks([0, 0.5, 1], labels=["0 opposite", "0.5", "1 same"])
    return fig, {"spearman_rho_vs_transcripts": r, "median_agreement": med, "bins": int(ok.sum())}


def main() -> None:
    args = _parse_args()
    out_dir = args.output_dir or args.cells.parent
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(args.cells)
    cells = cells[cells["in_fish"] & ~cells["saturated"] & ~cells["debris"]].copy()
    tx = f"{args.gene}_transcripts"
    shape = (int(cells.fish_y.max() // args.bin_px) + 1, int(cells.fish_x.max() // args.bin_px) + 1)
    maps, show = build_maps(cells, shape, args, args.min_cells)
    tx_name = f"{args.gene} transcripts per cell (Xenium)"

    extent = (0, shape[1] * args.bin_px, shape[0] * args.bin_px, 0)
    um = args.bin_px * args.sigma_bins * FISH_UM_PER_PX
    rows, figures = [], []
    for name, m in maps.items():
        fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
        cmap = "viridis" if name == tx_name else "magma"
        im = ax.imshow(m, extent=extent, cmap=cmap, vmin=np.nanpercentile(m, 1), vmax=np.nanpercentile(m, 99), interpolation="bilinear")
        ax.set_facecolor("#e8eaec")
        title = name
        if name != tx_name:
            ok = np.isfinite(m) & np.isfinite(maps[tx_name])
            r = spearmanr(m[ok], maps[tx_name][ok]).statistic
            rows.append({"fish_map": name, "cells": "all", "spearman_rho_vs_transcripts": r, "bins": int(ok.sum())})
            title += f"\nspatial Spearman rho vs {args.gene} transcripts = {r:.2f}"
        ax.set_title(f"{title}  (smoothed ~{um:.0f} µm)")
        ax.set_xlabel("FISH x (px)")
        ax.set_ylabel("FISH y (px)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        figures.append((f"spatial density {name}", fig))

    def unit(m: np.ndarray) -> np.ndarray:
        lo, hi = np.nanpercentile(m, [2, 98])
        return np.nan_to_num(np.clip((m - lo) / (hi - lo), 0, 1))

    for name in FISH_MAPS:
        rgb = np.dstack([unit(maps[name]), unit(maps[tx_name]), np.zeros(shape)])
        rgb[~show] = 0.92
        fig, (ax, key) = plt.subplots(1, 2, figsize=(12, 6.5), constrained_layout=True,
                                      gridspec_kw={"width_ratios": [6, 1]})
        ax.imshow(rgb, extent=extent, interpolation="bilinear")
        r = next(row["spearman_rho_vs_transcripts"] for row in rows if row["fish_map"] == name and row["cells"] == "all")
        ax.set_title(f"{name} vs {args.gene} transcripts\nspatial Spearman rho = {r:.2f}  (smoothed ~{um:.0f} µm)")
        ax.set_xlabel("FISH x (px)")
        ax.set_ylabel("FISH y (px)")
        ramp = np.linspace(0, 1, 64)
        red, green = np.meshgrid(ramp, ramp)
        key.imshow(np.dstack([red, green, np.zeros_like(red)]), origin="lower", extent=(0, 1, 0, 1))
        key.set_box_aspect(1)
        key.set_xticks([0, 1], ["low", "high"])
        key.set_yticks([0, 1], ["low", "high"])
        key.set_xlabel(f"{name}\n(FISH, red)", fontsize=9)
        key.set_ylabel(f"{args.gene} transcripts\n(Xenium, green)", fontsize=9)
        key.set_title("yellow = both high", fontsize=9)
        figures.append((f"spatial density overlay {name}", fig))

        fig, stat = agreement_figure(maps[name], maps[tx_name], extent, f"Agreement, all cells: {name} vs {args.gene} transcripts", um)
        next(row for row in rows if row["fish_map"] == name and row["cells"] == "all")["median_agreement"] = stat["median_agreement"]
        figures.append((f"spatial density agreement {name}", fig))

    if args.celltypes is not None:
        types = pd.read_csv(args.celltypes, usecols=["cell_id", "group"])
        cells = cells.merge(types, on="cell_id", how="left")
        for label, groups in (("malignant", args.malignant), ("non-malignant", args.non_malignant)):
            sub = cells[cells["group"].isin(groups)]
            sub_maps, _ = build_maps(sub, shape, args, args.min_cells_subset)
            for name in FISH_MAPS:
                fig, stat = agreement_figure(sub_maps[name], sub_maps[tx_name], extent,
                                             f"Agreement, Xenium {label} cells (n={len(sub):,}): {name} vs {args.gene} transcripts", um)
                rows.append({"fish_map": name, "cells": label} | stat)
                figures.append((f"spatial density agreement {label} {name}", fig))

    for title, fig in figures:
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        fig.savefig(plot_dir / f"cells_{slug}.png", dpi=130)
    stats = pd.DataFrame(rows)
    stats.to_csv(out_dir / "spatial_density_stats.csv", index=False)
    print(stats.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"plots in {plot_dir}: cells_spatial_density_*.png")


if __name__ == "__main__":
    main()
