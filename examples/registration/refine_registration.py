"""Refine a ``register_cells.py`` result to sub-pixel accuracy and report it.

The exhaustive search is limited by its rotation/scale step and by the search
canvas resolution, which typically leaves an offset of a few pixels that grows
across the field. This script measures that offset directly and removes it:

1. Warp the moving image into the reference frame with the manifest matrix.
2. Cut the overlap into blocks and measure the local shift in each block by
   phase correlation.
3. Fit one affine correction to all block shifts and fold it into the matrix.
4. Repeat until the measured offset stops improving.

Outputs (in ``--output-dir``, default: next to the manifest, in ``refine/``):
``refined_transform.json``, ``block_offsets.csv``, ``report.html`` and
``report.pdf``.

Run:
    uv run python examples/registration/refine_registration.py \
        --manifest outputs/.../registration.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation

Image.MAX_IMAGE_PIXELS = None

MATRIX_KEY = "ref_full_from_mov_full"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="registration.json from register_cells.py")
    parser.add_argument("--moving", type=Path, default=None, help="Override the moving image path in the manifest.")
    parser.add_argument("--reference", type=Path, default=None, help="Override the reference image path in the manifest.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <manifest dir>/refine")
    parser.add_argument("--block-size", type=int, default=512, help="Block size in reference pixels.")
    parser.add_argument("--max-shift", type=float, default=60.0, help="Reject blocks with a larger measured shift (px).")
    parser.add_argument("--min-corr", type=float, default=0.25, help="Reject blocks whose correlation after shifting is lower.")
    parser.add_argument("--max-iters", type=int, default=4)
    parser.add_argument("--tol", type=float, default=0.3, help="Stop when the median offset is below this (px).")
    parser.add_argument(
        "--reference-pixel-size", type=float, default=None,
        help="Reference pixel size in um, only used to also print offsets in um.",
    )
    return parser.parse_args()


@dataclass
class Measurement:
    matrix: np.ndarray  # 3x3 ref_full_from_mov_full used for this measurement
    blocks: np.ndarray  # (N, 6): x, y, dx, dy, corr_after, corr_before (all blocks)
    good: np.ndarray  # subset of blocks that passed the filters

    @property
    def magnitude(self) -> np.ndarray:
        return np.hypot(self.good[:, 2], self.good[:, 3])

    def stats(self) -> dict[str, float]:
        mag = self.magnitude
        return {
            "blocks_used": int(len(self.good)),
            "blocks_total": int(len(self.blocks)),
            "median_px": float(np.median(mag)),
            "p90_px": float(np.percentile(mag, 90)),
            "max_px": float(mag.max()),
            "median_corr": float(np.median(self.good[:, 5])),
        }


def _warp_to_reference(image: Image.Image, matrix: np.ndarray, size: tuple[int, int], resample) -> np.ndarray:
    inverse = np.linalg.inv(matrix)
    warped = image.transform(size, Image.AFFINE, tuple(inverse[:2].ravel()), resample=resample)
    return np.asarray(warped, dtype=np.float32)


def _prep(block: np.ndarray) -> np.ndarray:
    """Band-pass so nuclei dominate and slow illumination changes do not."""
    smooth = gaussian_filter(block, 1.5)
    return smooth - gaussian_filter(smooth, 25)


def measure(
    reference: np.ndarray,
    moving: Image.Image,
    matrix: np.ndarray,
    *,
    block: int,
    max_shift: float,
    min_corr: float,
) -> tuple[Measurement, np.ndarray]:
    h, w = reference.shape
    warped = _warp_to_reference(moving, matrix, (w, h), Image.BILINEAR)
    coverage = _warp_to_reference(Image.new("L", moving.size, 255), matrix, (w, h), Image.NEAREST)

    rows = []
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            sl = (slice(y, y + block), slice(x, x + block))
            if coverage[sl].mean() < 250:  # block not fully inside the moving image
                continue
            ref_b, mov_b = reference[sl], warped[sl]
            if ref_b.std() < 2 or mov_b.std() < 2:  # empty / background
                continue
            ref_p, mov_p = _prep(ref_b), _prep(mov_b)
            shift, _, _ = phase_cross_correlation(ref_p, mov_p, upsample_factor=10, normalization=None)
            dy, dx = float(shift[0]), float(shift[1])
            shifted = np.roll(mov_p, (int(round(dy)), int(round(dx))), (0, 1))
            corr_after = float(np.corrcoef(ref_p.ravel(), shifted.ravel())[0, 1])
            corr_before = float(np.corrcoef(ref_p.ravel(), mov_p.ravel())[0, 1])
            rows.append((x + block / 2, y + block / 2, dx, dy, corr_after, corr_before))

    blocks = np.array(rows, dtype=np.float64).reshape(-1, 6)
    keep = (blocks[:, 4] > min_corr) & (np.hypot(blocks[:, 2], blocks[:, 3]) < max_shift)
    return Measurement(matrix=matrix.copy(), blocks=blocks, good=blocks[keep]), warped


def fit_correction(good: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares affine mapping block centres to their corrected positions.

    Returns the 3x3 correction and the per-block residual left after it.
    One pass of outlier rejection keeps a few bad blocks from tilting the fit.
    """
    src = np.c_[good[:, 0], good[:, 1], np.ones(len(good))]
    dst = np.c_[good[:, 0] + good[:, 2], good[:, 1] + good[:, 3]]

    def _solve(mask: np.ndarray) -> np.ndarray:
        coef, *_ = np.linalg.lstsq(src[mask], dst[mask], rcond=None)
        return coef.T  # (2, 3)

    mask = np.ones(len(good), dtype=bool)
    affine = _solve(mask)
    resid = np.hypot(*(dst - src @ affine.T).T)
    cutoff = max(1.0, np.median(resid) + 4 * 1.4826 * np.median(np.abs(resid - np.median(resid))))
    mask = resid <= cutoff
    if mask.sum() >= 6:
        affine = _solve(mask)
    residual = dst - src @ affine.T
    return np.vstack([affine, [0.0, 0.0, 1.0]]), residual


def decompose(matrix: np.ndarray) -> dict[str, float]:
    return {
        "scale_x": float(np.hypot(matrix[0, 0], matrix[1, 0])),
        "scale_y": float(np.hypot(matrix[0, 1], matrix[1, 1])),
        "rotation_deg": float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))),
        "shift_x": float(matrix[0, 2]),
        "shift_y": float(matrix[1, 2]),
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _quiver_figure(first: Measurement, last: Measurement, shape_hw: tuple[int, int]) -> plt.Figure:
    arrow = 25
    vmax = max(1.0, float(np.percentile(first.magnitude, 98)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    for ax, m, title in (
        (axes[0], first, "Before refinement"),
        (axes[1], last, "After refinement"),
    ):
        g = m.good
        q = ax.quiver(
            g[:, 0], g[:, 1], g[:, 2], g[:, 3], np.hypot(g[:, 2], g[:, 3]),
            angles="xy", scale_units="xy", scale=1 / arrow, cmap="viridis", clim=(0, vmax),
        )
        ax.set_xlim(0, shape_hw[1])
        ax.set_ylim(shape_hw[0], 0)
        ax.set_aspect("equal")
        ax.set_title(f"{title}: median {np.median(m.magnitude):.2f} px (arrows x{arrow})")
        ax.set_xlabel("reference x (px)")
        ax.set_ylabel("reference y (px)")
        fig.colorbar(q, ax=ax, label="offset (reference px)", shrink=0.8)
    return fig


def _histogram_figure(first: Measurement, last: Measurement) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    top = max(1.0, float(first.magnitude.max()))
    bins = np.linspace(0, top, 40)
    ax.hist(first.magnitude, bins=bins, alpha=0.6, color="#c0392b", label="before")
    ax.hist(last.magnitude, bins=bins, alpha=0.6, color="#16808a", label="after")
    ax.set_xlabel("block offset (reference px)")
    ax.set_ylabel("blocks")
    ax.set_title("Distribution of block offsets")
    ax.legend()
    return fig


def _stretch(a: np.ndarray) -> np.ndarray:
    nz = a[a > 0]
    if nz.size == 0:
        return np.zeros_like(a)
    lo, hi = np.percentile(nz, (1, 99.8))
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1) ** 0.7


def _overlay(ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
    r, m = _stretch(ref)[..., None], _stretch(mov)[..., None]
    return np.clip(r * np.array([0.45, 0.80, 0.80]) + m * np.array([1.0, 0.15, 0.15]), 0, 1)


def _crops_figure(
    reference: np.ndarray, warped_before: np.ndarray, warped_after: np.ndarray, first: Measurement, crop: int
) -> plt.Figure:
    """Before/after overlays at the worst, median and best block of the first pass."""
    order = np.argsort(first.magnitude)
    picks = [("largest offset", order[-1]), ("median offset", order[len(order) // 2]), ("smallest offset", order[0])]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    h, w = reference.shape
    for col, (label, idx) in enumerate(picks):
        cx, cy = first.good[idx, 0], first.good[idx, 1]
        x0 = int(np.clip(cx - crop / 2, 0, w - crop))
        y0 = int(np.clip(cy - crop / 2, 0, h - crop))
        sl = (slice(y0, y0 + crop), slice(x0, x0 + crop))
        for row, (warped, when) in enumerate(((warped_before, "before"), (warped_after, "after"))):
            ax = axes[row, col]
            ax.imshow(_overlay(reference[sl], warped[sl]), interpolation="nearest")
            ax.set_title(f"{when}: {label} block, x={x0}, y={y0}")
            ax.axis("off")
    fig.suptitle("Reference in teal, moving in red; overlapping nuclei appear pink-white")
    return fig


def _png_data_uri(fig: plt.Figure, dpi: int = 110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


def _matrix_text(matrix: np.ndarray) -> str:
    return "\n".join("[ " + "  ".join(f"{v: .6f}" if abs(v) < 10 else f"{v: .3f}" for v in row) + " ]" for row in matrix[:2])


def _fmt_um(px: float, pixel_size: float | None) -> str:
    return f"{px:.2f} px" + (f" ({px * pixel_size:.2f} um)" if pixel_size else "")


def _summary_rows(history: list[Measurement], pixel_size: float | None) -> list[list[str]]:
    rows = []
    for i, m in enumerate(history):
        s = m.stats()
        label = "initial (register_cells)" if i == 0 else f"after pass {i}"
        rows.append([
            label,
            f"{s['blocks_used']} / {s['blocks_total']}",
            _fmt_um(s["median_px"], pixel_size),
            _fmt_um(s["p90_px"], pixel_size),
            _fmt_um(s["max_px"], pixel_size),
            f"{s['median_corr']:.3f}",
        ])
    return rows


SUMMARY_HEADER = ["Stage", "Blocks used", "Median offset", "90th percentile", "Largest", "Median correlation"]
STEPS = [
    "Warp the moving image into the reference frame with the matrix from register_cells.py.",
    "Cut the overlap into blocks and measure the local shift in each block by phase correlation (0.1 px precision).",
    "Drop blocks with low correlation or implausibly large shifts.",
    "Fit one affine correction to all block shifts by least squares and fold it into the matrix.",
    "Re-measure with the updated matrix and repeat until the median offset stops improving.",
]


def write_html(path: Path, ctx: dict) -> None:
    def table(header: list[str], rows: list[list[str]]) -> str:
        head = "".join(f"<th>{h}</th>" for h in header)
        body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    d0, d1 = ctx["decomp_initial"], ctx["decomp_refined"]
    decomp_rows = [
        ["Scale x", f"{d0['scale_x']:.5f}", f"{d1['scale_x']:.5f}"],
        ["Scale y", f"{d0['scale_y']:.5f}", f"{d1['scale_y']:.5f}"],
        ["Rotation (deg)", f"{d0['rotation_deg']:.4f}", f"{d1['rotation_deg']:.4f}"],
        ["Shift x (px)", f"{d0['shift_x']:.2f}", f"{d1['shift_x']:.2f}"],
        ["Shift y (px)", f"{d0['shift_y']:.2f}", f"{d1['shift_y']:.2f}"],
    ]
    steps = "".join(f"<li>{s}</li>" for s in STEPS)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Registration refinement report</title>
<style>
  body {{ font: 15px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #1c2328;
         background: #f6f7f8; margin: 0; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px 48px; }}
  h1 {{ font-size: 24px; margin: 0 0 4px; }} h2 {{ font-size: 18px; margin: 32px 0 8px; }}
  .meta {{ color: #5b6770; font-size: 13px; word-break: break-all; }}
  .headline {{ background: #fff; border-left: 4px solid #16808a; padding: 12px 16px; margin: 16px 0; }}
  table {{ border-collapse: collapse; background: #fff; width: 100%; font-size: 14px; }}
  th, td {{ border: 1px solid #dde1e4; padding: 6px 10px; text-align: left; }}
  th {{ background: #eef1f3; }}
  pre {{ background: #fff; border: 1px solid #dde1e4; padding: 10px 12px; overflow-x: auto; font-size: 13px; }}
  img {{ max-width: 100%; height: auto; background: #fff; border: 1px solid #dde1e4; }}
  .two {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
  .scroll {{ overflow-x: auto; }}
</style></head><body><main>
<h1>Registration refinement report</h1>
<div class="meta">Generated {ctx['timestamp']}<br>Manifest: {ctx['manifest']}<br>
Moving: {ctx['moving']}<br>Reference: {ctx['reference']}</div>
<div class="headline">{ctx['headline']}</div>

<h2>What was done</h2>
<ol>{steps}</ol>
<p>Block size {ctx['block']} px, {ctx['passes']} refinement pass(es). {ctx['stop_reason']}</p>

<h2>Offset at each stage</h2>
<div class="scroll">{table(SUMMARY_HEADER, ctx['summary_rows'])}</div>

<h2>Transformation matrix (moving pixel &rarr; reference pixel)</h2>
<div class="two">
  <div><strong>Initial</strong><pre>{_matrix_text(ctx['m_initial'])}</pre></div>
  <div><strong>Refined</strong><pre>{_matrix_text(ctx['m_refined'])}</pre></div>
</div>
<div class="scroll">{table(["", "Initial", "Refined"], decomp_rows)}</div>
<p>Total correction applied on top of the initial matrix:</p>
<pre>{_matrix_text(ctx['correction'])}</pre>

<h2>Offset field</h2>
<p>Each arrow is one block: direction and length show how far the moving image is from the reference there.</p>
<img src="{ctx['img_quiver']}" alt="Offset field before and after refinement">

<h2>Offset distribution</h2>
<img src="{ctx['img_hist']}" alt="Histogram of block offsets before and after">

<h2>Overlay crops</h2>
<img src="{ctx['img_crops']}" alt="Before and after overlays at three blocks">
</main></body></html>
"""
    path.write_text(html, encoding="utf-8")


def write_pdf(path: Path, ctx: dict, figures: list[plt.Figure]) -> None:
    d0, d1 = ctx["decomp_initial"], ctx["decomp_refined"]
    lines = [
        "Registration refinement report",
        "",
        f"Generated: {ctx['timestamp']}",
        f"Manifest:  {ctx['manifest']}",
        f"Moving:    {ctx['moving']}",
        f"Reference: {ctx['reference']}",
        "",
        ctx["headline_plain"],
        "",
        "What was done",
        *[f"  {i}. {s}" for i, s in enumerate(STEPS, 1)],
        f"  Block size {ctx['block']} px, {ctx['passes']} refinement pass(es). {ctx['stop_reason']}",
        "",
        "Offset at each stage",
        "  " + " | ".join(SUMMARY_HEADER),
        *["  " + " | ".join(r) for r in ctx["summary_rows"]],
        "",
        "Initial matrix (moving pixel -> reference pixel)",
        *["  " + ln for ln in _matrix_text(ctx["m_initial"]).splitlines()],
        "Refined matrix",
        *["  " + ln for ln in _matrix_text(ctx["m_refined"]).splitlines()],
        "Total correction",
        *["  " + ln for ln in _matrix_text(ctx["correction"]).splitlines()],
        "",
        "                 initial      refined",
        f"  scale x       {d0['scale_x']:.5f}     {d1['scale_x']:.5f}",
        f"  scale y       {d0['scale_y']:.5f}     {d1['scale_y']:.5f}",
        f"  rotation deg  {d0['rotation_deg']:.4f}      {d1['rotation_deg']:.4f}",
        f"  shift x px    {d0['shift_x']:.2f}      {d1['shift_x']:.2f}",
        f"  shift y px    {d0['shift_y']:.2f}      {d1['shift_y']:.2f}",
    ]
    with PdfPages(path) as pdf:
        page = plt.figure(figsize=(11, 8.5))
        page.text(0.05, 0.95, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=7.5, wrap=False)
        pdf.savefig(page)
        plt.close(page)
        for fig in figures:
            pdf.savefig(fig)


# --------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    manifest = json.loads(args.manifest.read_text())
    m_initial = np.array(manifest["transform"]["matrices"][MATRIX_KEY], dtype=np.float64)
    moving_path = args.moving or Path(manifest["sources"]["moving"]["path"])
    reference_path = args.reference or Path(manifest["sources"]["reference"]["path"])
    out_dir = args.output_dir or args.manifest.parent / "refine"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"moving:    {moving_path}")
    print(f"reference: {reference_path}")
    moving = Image.open(moving_path).convert("L")
    reference = np.asarray(Image.open(reference_path).convert("L"), dtype=np.float32)

    kwargs = dict(block=args.block_size, max_shift=args.max_shift, min_corr=args.min_corr)
    current, warped_before = measure(reference, moving, m_initial, **kwargs)
    if len(current.good) < 6:
        raise RuntimeError(
            f"Only {len(current.good)} usable blocks; the initial registration is probably wrong "
            "or the overlap is too small. Try a smaller --block-size."
        )
    history = [current]
    warped_after = warped_before
    stop_reason = f"Stopped at the pass limit ({args.max_iters})."
    print(f"initial: median offset {current.stats()['median_px']:.2f} px over {len(current.good)} blocks")

    for i in range(1, args.max_iters + 1):
        if current.stats()["median_px"] < args.tol:
            stop_reason = f"Stopped because the median offset is below {args.tol} px."
            break
        correction, _ = fit_correction(current.good)
        candidate, warped = measure(reference, moving, correction @ current.matrix, **kwargs)
        if len(candidate.good) < 6 or candidate.stats()["median_px"] >= current.stats()["median_px"]:
            stop_reason = "Stopped because another pass did not reduce the offset."
            break
        current, warped_after = candidate, warped
        history.append(current)
        print(f"pass {i}: median offset {current.stats()['median_px']:.2f} px over {len(current.good)} blocks")

    first, last = history[0], history[-1]
    m_refined = last.matrix
    correction_total = m_refined @ np.linalg.inv(m_initial)
    s0, s1 = first.stats(), last.stats()
    px = args.reference_pixel_size
    headline_plain = (
        f"Median offset went from {_fmt_um(s0['median_px'], px)} to {_fmt_um(s1['median_px'], px)}; "
        f"largest block offset from {_fmt_um(s0['max_px'], px)} to {_fmt_um(s1['max_px'], px)}."
    )
    if len(history) == 1:
        headline_plain = f"No refinement applied. Median offset is {_fmt_um(s0['median_px'], px)}."

    figures = [
        _quiver_figure(first, last, reference.shape),
        _histogram_figure(first, last),
        _crops_figure(reference, warped_before, warped_after, first, min(args.block_size, 512)),
    ]
    ctx = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "manifest": str(args.manifest),
        "moving": str(moving_path),
        "reference": str(reference_path),
        "headline": f"<strong>{headline_plain}</strong>",
        "headline_plain": headline_plain,
        "block": args.block_size,
        "passes": len(history) - 1,
        "stop_reason": stop_reason,
        "summary_rows": _summary_rows(history, px),
        "m_initial": m_initial,
        "m_refined": m_refined,
        "correction": correction_total,
        "decomp_initial": decompose(m_initial),
        "decomp_refined": decompose(m_refined),
        "img_quiver": _png_data_uri(figures[0]),
        "img_hist": _png_data_uri(figures[1]),
        "img_crops": _png_data_uri(figures[2]),
    }
    write_html(out_dir / "report.html", ctx)
    write_pdf(out_dir / "report.pdf", ctx, figures)
    for fig in figures:
        plt.close(fig)

    np.savetxt(
        out_dir / "block_offsets.csv",
        np.c_[last.blocks, (last.blocks[:, 4] > args.min_corr) & (np.hypot(last.blocks[:, 2], last.blocks[:, 3]) < args.max_shift)],
        delimiter=",", header="x,y,dx,dy,corr_after,corr_before,used", comments="", fmt="%.4f",
    )
    (out_dir / "refined_transform.json").write_text(json.dumps({
        "schema": "batchmatch.refined_transform",
        "source_manifest": str(args.manifest),
        "moving": str(moving_path),
        "reference": str(reference_path),
        "matrices": {
            f"{MATRIX_KEY}_initial": m_initial.tolist(),
            MATRIX_KEY: m_refined.tolist(),
            "correction": correction_total.tolist(),
        },
        "decomposition": {"initial": ctx["decomp_initial"], "refined": ctx["decomp_refined"]},
        "block_size": args.block_size,
        "passes": [m.stats() for m in history],
        "stop_reason": stop_reason,
    }, indent=2))

    print(headline_plain)
    print(stop_reason)
    print(f"refined matrix:\n{_matrix_text(m_refined)}")
    print(f"outputs: {out_dir}/report.html, report.pdf, refined_transform.json, block_offsets.csv")


if __name__ == "__main__":
    main()
