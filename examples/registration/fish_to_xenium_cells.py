"""Assign FISH signal to Xenium cells using a refined registration.

Takes the refined FISH -> Xenium-tile transform from ``refine_registration.py``
and, for every Xenium cell that falls inside the FISH image:

- measures each FISH probe channel inside the Xenium nucleus outline and in
  the rest of the cell (cell outline minus nucleus),
- detects candidate FISH spots and assigns each one to a nucleus, to the
  extranuclear part of a cell, or to no cell,
- joins the Xenium transcript counts for the requested genes.

Coordinate chain (verified against Xenium nucleus outlines):
FISH pixel -> tile pixel (refined matrix) -> full-resolution Xenium pixel
(tile offset, downscale, rotation) -> Xenium micrometres.

Outputs in ``--output-dir`` (default: ``<refined dir>/cells``):
``cells.csv``, ``spots.csv``, ``transforms.json``, ``report.html``,
``report.pdf``.

Needs pandas, pyarrow and h5py on top of the project environment:
    uv run --with pandas --with pyarrow --with h5py \
        python examples/registration/fish_to_xenium_cells.py \
        --refined outputs/.../refine/refined_transform.json \
        --fish-tiff /path/to/fish.tif --xenium-dir /path/to/xenium_output
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from skimage.filters import threshold_otsu

Image.MAX_IMAGE_PIXELS = None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refined", type=Path, required=True, help="refined_transform.json from refine_registration.py")
    p.add_argument("--fish-tiff", type=Path, required=True, help="Multi-channel FISH TIFF on the same pixel grid as the registered FISH image.")
    p.add_argument("--xenium-dir", type=Path, required=True, help="Xenium output folder (cells.parquet, *_boundaries.parquet, cell_feature_matrix.h5).")
    p.add_argument("--tile", type=Path, default=None, help="Tile PNG used as reference. Default: path stored in the refined transform.")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--egfr-channel", type=int, default=0, help="FISH TIFF channel of the EGFR probe (mCherry).")
    p.add_argument("--cep7-channel", type=int, default=1, help="FISH TIFF channel of the chr7 centromere probe (FITC).")
    p.add_argument("--dapi-channel", type=int, default=2)
    p.add_argument("--genes", nargs="+", default=["EGFR"], help="Xenium genes whose per-cell transcript counts are added.")
    p.add_argument("--tile-x0", type=int, default=None, help="Tile x offset in full-resolution Xenium pixels. Default: parsed from '_x<N>_' in the tile name.")
    p.add_argument("--tile-y0", type=int, default=None, help="Tile y offset in full-resolution Xenium pixels. Default: parsed from '_y<N>' in the tile name.")
    p.add_argument("--tile-scale", type=int, default=4, help="Downscale factor applied when the tile PNG was made.")
    p.add_argument("--tile-rotate", type=int, default=270, choices=(0, 270), help="Counter-clockwise rotation applied when the tile PNG was made.")
    p.add_argument("--xenium-pixel-size", type=float, default=0.2125, help="Full-resolution Xenium pixel size in um.")
    p.add_argument("--spot-threshold", type=float, default=6.0, help="Spot threshold in robust noise units of the spot-filtered image.")
    return p.parse_args()


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def um_from_tile_matrix(tile_wh: tuple[int, int], x0: int, y0: int, scale: int, rotate: int, pixel: float) -> np.ndarray:
    """Affine taking tile PNG pixel (X, Y) to Xenium micrometres."""
    w, _ = tile_wh
    k = pixel * scale
    if rotate == 0:
        return np.array([[k, 0, pixel * x0], [0, k, pixel * y0], [0, 0, 1.0]])
    # PIL ROTATE_270: scaled (x_s, y_s) -> tile (X, Y) = (H_s - y_s, x_s), and H_s is the tile width.
    return np.array([[0, k, pixel * x0], [-k, 0, pixel * (y0 + scale * w)], [0, 0, 1.0]])


def apply_affine(matrix: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2], matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]


# --------------------------------------------------------------------------
# Xenium tables
# --------------------------------------------------------------------------


def load_polygons(path: Path, cell_index: pd.Series, fish_from_um: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Return (label, vertices in FISH pixels) for every outline of the selected cells."""
    df = pd.read_parquet(path, columns=["cell_id", "vertex_x", "vertex_y", "label_id"])
    df = df[df["cell_id"].isin(cell_index.index)]
    fx, fy = apply_affine(fish_from_um, df["vertex_x"].to_numpy(), df["vertex_y"].to_numpy())
    labels = cell_index.reindex(df["cell_id"]).to_numpy()
    part = df["label_id"].to_numpy()
    # A new outline starts whenever the cell or the outline id changes.
    breaks = np.flatnonzero((labels[1:] != labels[:-1]) | (part[1:] != part[:-1])) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, len(df)]
    xy = np.c_[fx, fy]
    return [(int(labels[s]), xy[s:e]) for s, e in zip(starts, ends) if e - s >= 3]


def rasterize(polygons: list[tuple[int, np.ndarray]], shape_hw: tuple[int, int]) -> np.ndarray:
    img = Image.new("I", (shape_hw[1], shape_hw[0]), 0)
    draw = ImageDraw.Draw(img)
    for label, xy in polygons:
        draw.polygon([tuple(v) for v in xy], fill=label)
    return np.asarray(img, dtype=np.int32)


def gene_counts(h5_path: Path, genes: list[str]) -> pd.DataFrame:
    with h5py.File(h5_path, "r") as h:
        m = h["matrix"]
        names = [n.decode() for n in m["features/name"][:]]
        barcodes = [b.decode() for b in m["barcodes"][:]]
        indices, indptr, data = m["indices"][:], m["indptr"][:], m["data"][:]
    out = pd.DataFrame(index=pd.Index(barcodes, name="cell_id"))
    for gene in genes:
        if gene not in names:
            raise ValueError(f"Gene {gene!r} is not in the Xenium panel.")
        pos = np.flatnonzero(indices == names.index(gene))
        counts = np.zeros(len(barcodes), dtype=np.int32)
        counts[np.searchsorted(indptr, pos, side="right") - 1] = data[pos]
        out[f"{gene}_transcripts"] = counts
    return out


# --------------------------------------------------------------------------
# FISH measurements
# --------------------------------------------------------------------------


def region_stats(channel: np.ndarray, labels: np.ndarray, n: int, background: float, prefix: str) -> pd.DataFrame:
    idx = np.arange(1, n + 1)
    area = ndi.sum(np.ones_like(channel, dtype=np.float32), labels, idx)
    total = ndi.sum(channel, labels, idx)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(area > 0, total / area - background, np.nan)
    return pd.DataFrame({
        f"{prefix}_area_px": area.astype(np.int32),
        f"{prefix}_mean": mean,
        f"{prefix}_sum": np.where(area > 0, total - background * area, np.nan),
        f"{prefix}_max": np.where(area > 0, ndi.maximum(channel, labels, idx) - background, np.nan),
    })


def detect_spots(channel: np.ndarray, tissue: np.ndarray, threshold: float) -> pd.DataFrame:
    """Local maxima of a difference-of-Gaussians image above a robust noise threshold."""
    dog = ndi.gaussian_filter(channel, 0.8) - ndi.gaussian_filter(channel, 3.0)
    sample = dog[tissue][:: max(1, int(tissue.sum()) // 2_000_000)]
    noise = 1.4826 * np.median(np.abs(sample - np.median(sample)))
    peaks = (dog == ndi.maximum_filter(dog, size=3)) & (dog > threshold * noise)
    ys, xs = np.nonzero(peaks)
    return pd.DataFrame({
        "fish_x": xs, "fish_y": ys,
        "snr": dog[ys, xs] / noise,
        "raw": channel[ys, xs],
        "saturated": channel[ys, xs] >= 65535,
    })


# --------------------------------------------------------------------------
# Figures and report
# --------------------------------------------------------------------------


def _norm(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def crop_figure(fish, limits, nuc_polys, cell_polys, spots, centers, size=260) -> plt.Figure:
    fig, axes = plt.subplots(1, len(centers), figsize=(7 * len(centers), 7), constrained_layout=True)
    h, w = fish["dapi"].shape
    for ax, (title, cx, cy) in zip(np.atleast_1d(axes), centers):
        x0 = int(np.clip(cx - size / 2, 0, w - size))
        y0 = int(np.clip(cy - size / 2, 0, h - size))
        sl = (slice(y0, y0 + size), slice(x0, x0 + size))
        rgb = np.dstack([
            _norm(fish["egfr"][sl], *limits["egfr"]),
            _norm(fish["cep7"][sl], *limits["cep7"]),
            0.6 * _norm(fish["dapi"][sl], *limits["dapi"]) ** 0.6,
        ])
        ax.imshow(rgb, extent=(x0, x0 + size, y0 + size, y0), interpolation="nearest")
        for polys, color, lw in ((cell_polys, "#8a8f94", 0.5), (nuc_polys, "white", 0.8)):
            for _, xy in polys:
                if x0 - 40 < xy[0, 0] < x0 + size + 40 and y0 - 40 < xy[0, 1] < y0 + size + 40:
                    ax.plot(np.r_[xy[:, 0], xy[0, 0]], np.r_[xy[:, 1], xy[0, 1]], color=color, lw=lw)
        for name, marker, color in (("egfr", "o", "#ff5555"), ("cep7", "s", "#66ff66")):
            s = spots[(spots.channel == name) & spots.fish_x.between(x0, x0 + size) & spots.fish_y.between(y0, y0 + size)]
            ax.scatter(s.fish_x + 0.5, s.fish_y + 0.5, s=40, marker=marker, facecolors="none", edgecolors=color, linewidths=0.8)
        ax.set_xlim(x0, x0 + size)
        ax.set_ylim(y0 + size, y0)
        ax.set_title(f"{title} (FISH px x={x0}, y={y0})")
        ax.axis("off")
    fig.suptitle("EGFR probe red, CEP7 probe green, DAPI blue. White: Xenium nucleus outline, grey: cell outline. "
                 "Circles: EGFR spots, squares: CEP7 spots.")
    return fig


def map_figure(cells: pd.DataFrame, gene: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    panels = (
        ("egfr_nuc_mean", "EGFR FISH, nuclear mean intensity", "magma"),
        (f"{gene}_transcripts", f"{gene} transcripts per cell (Xenium)", "viridis"),
    )
    for ax, (col, title, cmap) in zip(axes, panels):
        v = cells[col].to_numpy(dtype=float)
        sc = ax.scatter(cells.fish_x, cells.fish_y, c=v, s=1.5, cmap=cmap, vmin=0, vmax=np.nanpercentile(v, 99), linewidths=0)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("FISH x (px)")
        ax.set_ylabel("FISH y (px)")
        fig.colorbar(sc, ax=ax, shrink=0.8)
    return fig


def scatter_figure(cells: pd.DataFrame, gene: str) -> tuple[plt.Figure, dict[str, float]]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    y = np.log1p(cells[f"{gene}_transcripts"].to_numpy(dtype=float))
    rho = {}
    for ax, (col, label) in zip(axes, (
        ("egfr_nuc_mean", "EGFR FISH nuclear mean"),
        ("egfr_cell_sum", "EGFR FISH whole-cell integrated"),
        ("cep7_nuc_mean", "CEP7 FISH nuclear mean (control)"),
    )):
        x = cells[col].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        r = spearmanr(x[ok], y[ok]).statistic
        rho[col] = float(r)
        lo, hi = np.nanpercentile(x[ok], [0.5, 99.5])
        ax.hexbin(np.clip(x[ok], lo, hi), y[ok], gridsize=60, bins="log", cmap="viridis")
        ax.set_xlabel(f"{label} (background-subtracted)")
        ax.set_ylabel(f"log(1 + {gene} transcripts)")
        ax.set_title(f"Spearman rho = {r:.3f} (n = {ok.sum():,})")
    return fig, rho


def _uri(fig: plt.Figure, dpi: int = 100) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def write_reports(out_dir: Path, summary: list[tuple[str, str]], notes: list[str], figures: list[tuple[str, plt.Figure]]) -> None:
    rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in summary)
    figs = "".join(f"<h2>{t}</h2><img src='{_uri(f)}' alt='{t}'>" for t, f in figures)
    note_html = "".join(f"<li>{n}</li>" for n in notes)
    (out_dir / "report.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>FISH to Xenium cells</title>
<style>
 body {{ font: 15px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #1c2328; background: #f6f7f8; margin: 0; }}
 main {{ max-width: 1150px; margin: 0 auto; padding: 24px 16px 48px; }}
 h1 {{ font-size: 24px; margin: 0 0 12px; }} h2 {{ font-size: 18px; margin: 32px 0 8px; }}
 table {{ border-collapse: collapse; background: #fff; font-size: 14px; }}
 th, td {{ border: 1px solid #dde1e4; padding: 6px 10px; text-align: left; }} th {{ background: #eef1f3; }}
 img {{ max-width: 100%; height: auto; background: #fff; border: 1px solid #dde1e4; }}
 .scroll {{ overflow-x: auto; }}
</style></head><body><main>
<h1>FISH signal assigned to Xenium cells</h1>
<div class="scroll"><table>{rows}</table></div>
<h2>Notes</h2><ul>{note_html}</ul>
{figs}
</main></body></html>""", encoding="utf-8")
    with PdfPages(out_dir / "report.pdf") as pdf:
        page = plt.figure(figsize=(11, 8.5))
        text = ["FISH signal assigned to Xenium cells", ""] + [f"{k}: {v}" for k, v in summary] + ["", "Notes"] + [f"- {n}" for n in notes]
        page.text(0.05, 0.95, "\n".join(text), va="top", family="monospace", fontsize=8, wrap=True)
        pdf.savefig(page)
        plt.close(page)
        for _, fig in figures:
            pdf.savefig(fig)


# --------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    refined = json.loads(args.refined.read_text())
    tile_from_fish = np.array(refined["matrices"]["ref_full_from_mov_full"], dtype=np.float64)
    tile_path = args.tile or Path(refined["reference"])
    out_dir = args.output_dir or args.refined.parent / "cells"
    out_dir.mkdir(parents=True, exist_ok=True)

    x0, y0 = args.tile_x0, args.tile_y0
    if x0 is None or y0 is None:
        m = re.search(r"_x(\d+)_y(\d+)", tile_path.name)
        if m is None:
            raise ValueError("Cannot parse the tile offset from its name; pass --tile-x0 and --tile-y0.")
        x0, y0 = int(m.group(1)), int(m.group(2))
    with Image.open(tile_path) as im:
        tile_wh = im.size
    um_from_tile = um_from_tile_matrix(tile_wh, x0, y0, args.tile_scale, args.tile_rotate, args.xenium_pixel_size)
    um_from_fish = um_from_tile @ tile_from_fish
    fish_from_um = np.linalg.inv(um_from_fish)
    fish_pixel_um = float(np.sqrt(abs(np.linalg.det(um_from_fish[:2, :2]))))
    print(f"tile offset x0={x0}, y0={y0}; FISH pixel = {fish_pixel_um:.4f} um")

    stack = tifffile.imread(args.fish_tiff)
    fish = {
        "egfr": stack[args.egfr_channel].astype(np.float32),
        "cep7": stack[args.cep7_channel].astype(np.float32),
        "dapi": stack[args.dapi_channel].astype(np.float32),
    }
    del stack
    h, w = fish["dapi"].shape

    # Cells whose centroid is inside the FISH image.
    cells = pd.read_parquet(args.xenium_dir / "cells.parquet")
    fx, fy = apply_affine(fish_from_um, cells["x_centroid"].to_numpy(), cells["y_centroid"].to_numpy())
    inside = (fx >= 0) & (fx < w) & (fy >= 0) & (fy < h)
    cells = cells.loc[inside, ["cell_id", "x_centroid", "y_centroid", "transcript_counts", "cell_area", "nucleus_area"]].copy()
    cells["fish_x"], cells["fish_y"] = fx[inside], fy[inside]
    cells = cells.reset_index(drop=True)
    n = len(cells)
    cell_index = pd.Series(np.arange(1, n + 1), index=cells["cell_id"])
    print(f"Xenium cells inside the FISH image: {n:,}")

    nuc_polys = load_polygons(args.xenium_dir / "nucleus_boundaries.parquet", cell_index, fish_from_um)
    cell_polys = load_polygons(args.xenium_dir / "cell_boundaries.parquet", cell_index, fish_from_um)
    nuc_lab = rasterize(nuc_polys, (h, w))
    cell_lab = rasterize(cell_polys, (h, w))
    cyto_lab = np.where(nuc_lab == 0, cell_lab, 0)

    tissue = ndi.binary_dilation(cell_lab > 0, iterations=10)
    spot_tables = []
    for name in ("egfr", "cep7"):
        channel = fish[name]
        background = float(np.median(channel[cell_lab == 0]))
        for labels, region in ((nuc_lab, "nuc"), (cyto_lab, "cyto"), (cell_lab, "cell")):
            cells = pd.concat([cells, region_stats(channel, labels, n, background, f"{name}_{region}")], axis=1)
        cells[f"{name}_background"] = background

        spots = detect_spots(channel, tissue, args.spot_threshold)
        spots["channel"] = name
        in_nuc = nuc_lab[spots.fish_y, spots.fish_x]
        in_cell = cell_lab[spots.fish_y, spots.fish_x]
        label = np.where(in_nuc > 0, in_nuc, in_cell)
        spots["location"] = np.where(in_nuc > 0, "nucleus", np.where(in_cell > 0, "extranuclear", "outside_cells"))
        spots["cell_id"] = np.where(label > 0, cells["cell_id"].to_numpy()[np.maximum(label, 1) - 1], "")
        spots["x_um"], spots["y_um"] = apply_affine(um_from_fish, spots.fish_x + 0.5, spots.fish_y + 0.5)
        for loc, col in (("nucleus", "spots_nuc"), ("extranuclear", "spots_extranuc")):
            counts = np.bincount(label[(spots.location == loc).to_numpy()], minlength=n + 1)[1:]
            cells[f"{name}_{col}"] = counts
        spot_tables.append(spots)
        print(f"{name}: background {background:.0f}, {len(spots):,} spots "
              f"({(spots.location == 'nucleus').sum():,} nuclear, {(spots.location == 'extranuclear').sum():,} extranuclear, "
              f"{(spots.location == 'outside_cells').sum():,} outside cells)")
    spots = pd.concat(spot_tables, ignore_index=True)

    # Is the cell actually present in the FISH image, and is it free of saturated debris?
    dapi_background = float(np.median(fish["dapi"][cell_lab == 0]))
    cells["dapi_nuc_mean"] = region_stats(fish["dapi"], nuc_lab, n, dapi_background, "dapi_nuc")["dapi_nuc_mean"]
    dapi = cells["dapi_nuc_mean"].dropna()
    dapi_cut = float(threshold_otsu(np.clip(dapi, None, dapi.quantile(0.995)).to_numpy()))
    cells["in_fish"] = cells["dapi_nuc_mean"] > dapi_cut
    idx = np.arange(1, n + 1)
    cells["saturated"] = (
        np.maximum(ndi.maximum(fish["egfr"], cell_lab, idx), ndi.maximum(fish["cep7"], cell_lab, idx)) >= 65535
    )

    # Drop the columns that duplicate each other, add ratios and transcripts.
    cells = cells.drop(columns=[c for c in cells.columns if c.endswith("_cell_area_px") or c.endswith("_cyto_area_px")] + ["cep7_nuc_area_px"])
    cells = cells.rename(columns={"egfr_nuc_area_px": "nucleus_area_fish_px"})
    with np.errstate(invalid="ignore", divide="ignore"):
        cells["egfr_cep7_nuc_ratio"] = cells["egfr_nuc_mean"] / cells["cep7_nuc_mean"].where(cells["cep7_nuc_mean"] > 0)
    cells = cells.merge(gene_counts(args.xenium_dir / "cell_feature_matrix.h5", args.genes), left_on="cell_id", right_index=True, how="left")
    measured = cells[cells["nucleus_area_fish_px"] > 0]
    usable = measured[measured["in_fish"] & ~measured["saturated"]]

    cells.to_csv(out_dir / "cells.csv", index=False, float_format="%.4f")
    spots.to_csv(out_dir / "spots.csv", index=False, float_format="%.3f")
    (out_dir / "transforms.json").write_text(json.dumps({
        "um_from_fish_pixel": um_from_fish.tolist(),
        "fish_pixel_from_um": fish_from_um.tolist(),
        "um_from_tile_pixel": um_from_tile.tolist(),
        "tile_pixel_from_fish_pixel": tile_from_fish.tolist(),
        "fish_pixel_size_um": fish_pixel_um,
        "tile": {"path": str(tile_path), "x0": x0, "y0": y0, "scale": args.tile_scale, "rotate": args.tile_rotate},
    }, indent=2))

    gene = args.genes[0]
    limits = {k: tuple(np.percentile(v[::4, ::4][cell_lab[::4, ::4] > 0], q)) for k, v, q in (
        ("egfr", fish["egfr"], (20, 99.9)), ("cep7", fish["cep7"], (20, 99.9)), ("dapi", fish["dapi"], (5, 99.5)))}
    top = usable.nlargest(max(1, len(usable) // 200), "egfr_nuc_mean")
    centers = [
        ("Centre of FISH image", w / 2, h / 2),
        ("Around a high EGFR-FISH nucleus", float(top.fish_x.median()), float(top.fish_y.median())),
        ("Around the strongest EGFR-FISH nucleus", float(top.fish_x.iloc[0]), float(top.fish_y.iloc[0])),
    ]
    scatter, rho = scatter_figure(usable, gene)
    figures = [
        ("Spatial pattern: FISH signal vs Xenium transcripts", map_figure(usable, gene)),
        (f"Per-cell FISH signal vs {gene} transcripts", scatter),
        ("Example regions with Xenium outlines on the FISH image", crop_figure(fish, limits, nuc_polys, cell_polys, spots, centers)),
    ]
    summary = [
        ("FISH TIFF", str(args.fish_tiff)),
        ("Tile", f"{tile_path.name} (x0={x0}, y0={y0}, scale {args.tile_scale}, rotated {args.tile_rotate})"),
        ("FISH pixel size", f"{fish_pixel_um:.4f} um"),
        ("Xenium cells inside FISH image", f"{n:,}"),
        ("Cells with a measured nucleus", f"{len(measured):,}"),
        ("Of those, present in FISH (nuclear DAPI above cut)", f"{int(measured['in_fish'].sum()):,} ({measured['in_fish'].mean():.0%}), cut = {dapi_cut:.0f}"),
        ("Of those, touching saturated signal (debris)", f"{int(measured['saturated'].sum()):,}"),
        ("Usable cells (present, not saturated)", f"{len(usable):,}"),
        ("EGFR spots (nuclear / extranuclear / outside cells)", " / ".join(f"{(spots[spots.channel == 'egfr'].location == k).sum():,}" for k in ("nucleus", "extranuclear", "outside_cells"))),
        ("CEP7 spots (nuclear / extranuclear / outside cells)", " / ".join(f"{(spots[spots.channel == 'cep7'].location == k).sum():,}" for k in ("nucleus", "extranuclear", "outside_cells"))),
        (f"Spearman, EGFR FISH nuclear mean vs {gene} transcripts", f"{rho['egfr_nuc_mean']:.3f}"),
        (f"Spearman, CEP7 FISH nuclear mean vs {gene} transcripts", f"{rho['cep7_nuc_mean']:.3f}"),
    ]
    notes = [
        "Cell and nucleus outlines are Xenium's own segmentation, mapped onto the FISH image; no segmentation is done on the FISH image.",
        "Intensities are background-subtracted using the median of pixels outside all cells.",
        f"Spots are local maxima of a spot-enhancing filter above {args.spot_threshold:g} robust noise units; at {fish_pixel_um:.2f} um per pixel "
        "they are candidates, not a locus count.",
        "'Extranuclear' means inside the Xenium cell outline but outside its nucleus outline.",
        "A cell counts as present in FISH when the FISH DAPI signal inside its Xenium nucleus outline is above an Otsu cut; "
        "maps and correlations use only present, non-saturated cells.",
    ]
    write_reports(out_dir, summary, notes, figures)
    for k, v in summary:
        print(f"{k}: {v}")
    print(f"outputs in {out_dir}: cells.csv, spots.csv, transforms.json, report.html, report.pdf")


if __name__ == "__main__":
    main()
