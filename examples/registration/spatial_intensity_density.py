"""Whole-slide EGFR FISH signal density vs Xenium EGFR transcript density.

Pixel-level, no cells and no spot calling: every EGFR FISH pixel in the
tissue counts, inside or outside nuclei. Both modalities are put on the same
grid in FISH pixels and smoothed the same way:

- FISH: background-subtracted EGFR channel summed per grid bin, divided by
  the tissue area in that bin (signal per tissue pixel). Also EGFR/CEP7 =
  summed EGFR / summed CEP7 signal per bin, which cancels staining strength,
  section thickness and cell density.
- Xenium: EGFR transcripts (qv >= --min-qv) from ``transcripts.parquet``,
  mapped into FISH pixels with ``transforms.json`` from
  ``fish_to_xenium_cells.py``, counted per bin and divided by the same tissue
  area.

Tissue = smoothed DAPI above an Otsu threshold. Background = median of
non-tissue pixels per channel. Debris = bright connected regions much larger
than a nucleus, removed from both FISH and Xenium (same rule idea as
``fish_to_xenium_cells.py``, without nuclei). Bins with less than
``--min-tissue`` tissue fraction are left out.

With ``--celltypes``, the same maps are rebuilt per group: FISH pixels inside
the Xenium cell outlines of malignant (or non-malignant) cells, and the EGFR
transcripts Xenium assigned to those cells, each divided by that group's
area in the bin. Agreement is ranked within each group separately.

Outputs under ``<out>/plots``: ``intensity_density_*.png`` and
``intensity_density_stats.csv``.

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
    uv run --no-sync python examples/registration/spatial_intensity_density.py \
        --cells-dir outputs/.../refine/cells \
        --fish-tiff /path/to/fish.tif \
        --xenium-dir /path/to/xenium_output \
        --celltypes /path/to/annotations/p46p_celltype.csv   # optional: malignant / non-malignant maps
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tifffile
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from PIL import Image, ImageDraw
from skimage.filters import threshold_otsu

MALIGNANT = ["AC-like", "MES-like", "OPC-like", "Cycling"]
NON_MALIGNANT = ["Endothelial", "Pericyte", "Myeloid", "Neuron", "Oligodendrocyte", "T cell"]
FISH_MAPS = ("EGFR FISH signal density", "EGFR/CEP7 FISH signal")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells-dir", type=Path, required=True, help="cells/ folder with transforms.json")
    p.add_argument("--fish-tiff", type=Path, required=True)
    p.add_argument("--xenium-dir", type=Path, required=True, help="folder with transcripts.parquet")
    p.add_argument("--gene", default="EGFR")
    p.add_argument("--egfr-channel", type=int, default=0)
    p.add_argument("--cep7-channel", type=int, default=1)
    p.add_argument("--dapi-channel", type=int, default=2)
    p.add_argument("--min-qv", type=float, default=20)
    p.add_argument("--bin-px", type=int, default=50, help="grid bin in FISH px (default 50 = ~24 um)")
    p.add_argument("--sigma-bins", type=float, default=2.0, help="Gaussian smoothing in bins (default 2 = ~50 um)")
    p.add_argument("--min-tissue", type=float, default=0.3, help="min smoothed tissue fraction for a bin to count")
    p.add_argument("--debris-factor", type=float, default=6.0, help="debris = smoothed signal > factor x median tissue signal")
    p.add_argument("--debris-min-px", type=int, default=2000, help="... in connected regions larger than this (px)")
    p.add_argument("--celltypes", type=Path, default=None,
                   help="CSV with columns cell_id,group; adds agreement maps inside malignant and non-malignant cells")
    p.add_argument("--malignant", nargs="+", default=MALIGNANT, help="groups counted as malignant")
    p.add_argument("--non-malignant", nargs="+", default=NON_MALIGNANT, help="groups counted as non-malignant")
    p.add_argument("--min-group-area", type=float, default=0.05,
                   help="min smoothed fraction of a bin covered by that group's cells (sparser groups, default 0.05)")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def bin_sum(img: np.ndarray, b: int) -> np.ndarray:
    h, w = (img.shape[0] // b) * b, (img.shape[1] // b) * b
    return img[:h, :w].reshape(h // b, b, w // b, b).sum(axis=(1, 3))


def debris_mask(channel: np.ndarray, tissue: np.ndarray, factor: float, min_px: int) -> np.ndarray:
    smooth = ndi.gaussian_filter(channel, 2.0)
    level = float(np.median(smooth[tissue][::16]))
    labels, _ = ndi.label(smooth > factor * max(level, 1.0))
    sizes = np.bincount(labels.ravel())
    big = np.flatnonzero(sizes > min_px)
    return ndi.binary_dilation(np.isin(labels, big[big > 0]), iterations=10)


def cell_mask(path: Path, ids: set[str], fish_from_um: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """Pixels inside the Xenium cell outlines of the given cells, in the FISH frame."""
    df = pd.read_parquet(path, columns=["cell_id", "vertex_x", "vertex_y"])
    fx = fish_from_um[0, 0] * df.vertex_x + fish_from_um[0, 1] * df.vertex_y + fish_from_um[0, 2]
    fy = fish_from_um[1, 0] * df.vertex_x + fish_from_um[1, 1] * df.vertex_y + fish_from_um[1, 2]
    near = (fx > -50) & (fx < shape_hw[1] + 50) & (fy > -50) & (fy < shape_hw[0] + 50)
    keep_cells = set(df.loc[near, "cell_id"].astype(str)) & ids
    df = df[df["cell_id"].astype(str).isin(keep_cells)]
    fx, fy = fx[df.index], fy[df.index]
    img = Image.new("1", (shape_hw[1], shape_hw[0]), 0)
    draw = ImageDraw.Draw(img)
    xy = np.c_[fx.to_numpy(), fy.to_numpy()]
    cid = df["cell_id"].astype(str).to_numpy()
    starts = np.r_[0, np.flatnonzero(cid[1:] != cid[:-1]) + 1, len(cid)]
    for s, e in zip(starts[:-1], starts[1:]):
        if e - s >= 3 and cid[s] in keep_cells:
            draw.polygon([tuple(v) for v in xy[s:e]], fill=1)
    return np.asarray(img, dtype=bool)


def main() -> None:
    args = _parse_args()
    out_dir = args.output_dir or args.cells_dir
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fish_from_um = np.array(json.loads((args.cells_dir / "transforms.json").read_text())["fish_pixel_from_um"])
    fish_um = float(json.loads((args.cells_dir / "transforms.json").read_text())["fish_pixel_size_um"])

    stack = tifffile.imread(args.fish_tiff)
    egfr = stack[args.egfr_channel].astype(np.float32)
    cep7 = stack[args.cep7_channel].astype(np.float32)
    dapi = stack[args.dapi_channel].astype(np.float32)
    del stack
    h, w = dapi.shape

    dapi_s = ndi.gaussian_filter(dapi[::4, ::4], 4)
    tissue = np.repeat(np.repeat(dapi_s > threshold_otsu(dapi_s), 4, 0), 4, 1)[:h, :w]
    tissue = ndi.binary_closing(tissue, iterations=8)
    for ch in (egfr, cep7):
        ch -= float(np.median(ch[~tissue][::16]))
        np.clip(ch, 0, None, out=ch)
    debris = debris_mask(egfr, tissue, args.debris_factor, args.debris_min_px) | debris_mask(cep7, tissue, args.debris_factor, args.debris_min_px)
    good = tissue & ~debris
    print(f"tissue {tissue.mean():.1%} of the image, debris {debris.mean():.2%}")

    b = args.bin_px
    smooth = lambda a: ndi.gaussian_filter(a, args.sigma_bins)
    shape = (h // b, w // b)

    cols = ["feature_name", "x_location", "y_location", "qv", "cell_id"]
    tx = pq.read_table(args.xenium_dir / "transcripts.parquet", columns=cols,
                       filters=[("qv", ">=", args.min_qv)]).to_pandas()
    names = tx["feature_name"].astype(str).str.removeprefix("b'").str.removesuffix("'")
    tx = tx[names == args.gene]
    fx = fish_from_um[0, 0] * tx.x_location + fish_from_um[0, 1] * tx.y_location + fish_from_um[0, 2]
    fy = fish_from_um[1, 0] * tx.x_location + fish_from_um[1, 1] * tx.y_location + fish_from_um[1, 2]
    ix, iy = np.floor(fx.to_numpy()).astype(int), np.floor(fy.to_numpy()).astype(int)
    inside = (ix >= 0) & (ix < shape[1] * b) & (iy >= 0) & (iy < shape[0] * b)
    ix, iy = ix[inside], iy[inside]
    tx_cell = tx["cell_id"].astype(str).str.removeprefix("b'").str.removesuffix("'").reset_index(drop=True)[inside].reset_index(drop=True)
    print(f"{args.gene} transcripts in FISH tissue: {good[iy, ix].sum():,} of {inside.sum():,} inside the FISH frame")

    tx_name = f"{args.gene} transcript density (Xenium)"

    def compute_maps(region: np.ndarray, tx_keep: np.ndarray, min_frac: float) -> dict[str, np.ndarray]:
        """Smoothed maps restricted to ``region`` pixels and the ``tx_keep`` transcripts."""
        area = smooth(bin_sum(region.astype(np.float32), b))
        egfr_sum = smooth(bin_sum(np.where(region, egfr, 0), b))
        cep7_sum = smooth(bin_sum(np.where(region, cep7, 0), b))
        keep = tx_keep & region[iy, ix]
        counts = np.zeros(shape)
        np.add.at(counts, (iy[keep] // b, ix[keep] // b), 1)
        valid = area >= min_frac * b * b
        with np.errstate(invalid="ignore", divide="ignore"):
            maps = {
                "EGFR FISH signal density": egfr_sum / area,
                "EGFR/CEP7 FISH signal": egfr_sum / cep7_sum,
                tx_name: smooth(counts) / area / (fish_um ** 2) * 100,
            }
        for m in maps.values():
            m[~valid] = np.nan
        return maps

    maps = compute_maps(good, np.ones(len(ix), dtype=bool), args.min_tissue)
    units = {"EGFR FISH signal density": "background-subtracted signal per tissue pixel",
             "EGFR/CEP7 FISH signal": "summed EGFR / summed CEP7 signal",
             tx_name: "transcripts per 100 µm² of tissue"}

    extent = (0, shape[1] * b, shape[0] * b, 0)
    um = b * args.sigma_bins * fish_um
    rows, figures = [], []
    for name, m in maps.items():
        fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
        ax.set_facecolor("#e8eaec")
        im = ax.imshow(m, extent=extent, cmap="viridis" if name == tx_name else "magma",
                       vmin=np.nanpercentile(m, 1), vmax=np.nanpercentile(m, 99), interpolation="bilinear")
        title = name
        if name != tx_name:
            ok = np.isfinite(m) & np.isfinite(maps[tx_name])
            r = spearmanr(m[ok], maps[tx_name][ok]).statistic
            title += f"\nspatial Spearman rho vs {args.gene} transcript density = {r:.2f}"
        ax.set_title(f"{title}  (smoothed ~{um:.0f} µm)")
        ax.set_xlabel("FISH x (px)")
        ax.set_ylabel("FISH y (px)")
        fig.colorbar(im, ax=ax, shrink=0.8).set_label(units[name])
        figures.append((f"intensity density {name}", fig))

    def agreement(maps: dict[str, np.ndarray], name: str, label: str) -> None:
        """Percentile-rank both maps across bins; agreement = 1 - |rank difference|."""
        ok = np.isfinite(maps[name]) & np.isfinite(maps[tx_name])
        pf = pd.Series(maps[name][ok]).rank(pct=True).to_numpy()
        pt = pd.Series(maps[tx_name][ok]).rank(pct=True).to_numpy()
        agree = np.full(shape, np.nan)
        agree[ok] = 1 - np.abs(pf - pt)
        r = spearmanr(maps[name][ok], maps[tx_name][ok]).statistic
        med = float(np.median(agree[ok]))
        rows.append({"fish_map": name, "cells": label, "spearman_rho_vs_transcript_density": r,
                     "median_agreement": med, "bins": int(ok.sum())})
        fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
        ax.set_facecolor("#e8eaec")
        im = ax.imshow(agree, extent=extent, cmap="Blues", vmin=0, vmax=1, interpolation="bilinear")
        who = "all tissue" if label == "all" else f"inside Xenium {label} cells"
        ax.set_title(f"Agreement, {who}: {name} vs {args.gene} transcript density\n"
                     f"median agreement = {med:.2f}, spatial Spearman rho = {r:.2f}  (smoothed ~{um:.0f} µm)")
        ax.set_xlabel("FISH x (px)")
        ax.set_ylabel("FISH y (px)")
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label("agreement = 1 − |FISH percentile − transcript percentile|")
        cb.set_ticks([0, 0.5, 1], labels=["0 opposite", "0.5", "1 same"])
        slug = "" if label == "all" else f"{label} "
        figures.append((f"intensity density agreement {slug}{name}", fig))

    for name in FISH_MAPS:
        agreement(maps, name, "all")

    if args.celltypes is not None:
        types = pd.read_csv(args.celltypes, usecols=["cell_id", "group"])
        for label, groups in (("malignant", args.malignant), ("non-malignant", args.non_malignant)):
            ids = set(types.loc[types["group"].isin(groups), "cell_id"].astype(str))
            region = good & cell_mask(args.xenium_dir / "cell_boundaries.parquet", ids, fish_from_um, (h, w))
            print(f"{label}: {len(ids):,} annotated cells, {region.mean():.1%} of the FISH image")
            sub = compute_maps(region, tx_cell.isin(ids).to_numpy(), args.min_group_area)
            for name in FISH_MAPS:
                agreement(sub, name, label)

    for title, fig in figures:
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        fig.savefig(plot_dir / f"cells_{slug}.png", dpi=130)
    stats = pd.DataFrame(rows)
    stats.to_csv(out_dir / "intensity_density_stats.csv", index=False)
    print(stats.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"plots in {plot_dir}: cells_intensity_density_*.png")


if __name__ == "__main__":
    main()
