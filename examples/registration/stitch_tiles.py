"""Stitch per-tile transforms into a single smooth registration and warp.

Consumes the ``tile_transforms.json`` manifest produced by
``register_tiles.py`` and blends the per-tile affine transforms into one
smoothly-varying (piecewise-affine) deformation field over the reference
canvas.  The moving image is then resampled through that field to produce
a stitched, registered result.

Each tile contributes its local ``matrix_ref_full_from_mov_full`` affine,
weighted by a Gaussian of the distance from the output pixel to the
tile's reference-space center, so neighbouring tiles blend seamlessly
instead of showing seams.

The stitched output is produced at full resolution by default
(``--output-downsample 1``); pass a larger downsample for a quick
low-resolution preview.  The output canvas is iterated in tiles so the
sampling grid is never materialized for the whole canvas and memory
stays bounded regardless of canvas size.

Two previews are written: a red/green two-color overlay
(reference=green, moving=red, overlap=yellow) that is easy to read at any
scale, plus the classic gray+red alpha overlay and a checkerboard.

Run:
    uv run python examples/registration/stitch_tiles.py \
        --manifest outputs/register_tiles/tile_transforms.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from batchmatch.io import (
    SourceInfo,
    TiffExportConfig,
    load_image,
    save_tiff,
)
from batchmatch.view.config import (
    CheckerboardSpec,
    DisplaySpec,
    EdgeOverlaySpec,
    OverlaySpec,
)
from batchmatch.view.display import show_comparison


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "outputs" / "register_tiles" / "tile_transforms.json",
        help="tile_transforms.json produced by register_tiles.py.",
    )
    parser.add_argument(
        "--output-downsample",
        type=int,
        default=1,
        help=(
            "Downsample of the stitched output canvas (reference full-res / this). "
            "Default 1 (full resolution). Set >1 for a smaller, faster preview product."
        ),
    )
    parser.add_argument(
        "--moving-downsample",
        type=int,
        default=1,
        help=(
            "Downsample at which the moving image is loaded for resampling. "
            "Default 1 (full resolution). Increase if the moving image does not "
            "fit in memory."
        ),
    )
    parser.add_argument(
        "--blend-sigma",
        type=float,
        default=0.6,
        help=(
            "Gaussian blend width as a fraction of the tile-center spacing. "
            "Larger = smoother field, smaller = more locally exact per tile."
        ),
    )
    parser.add_argument(
        "--include-low-score",
        action="store_true",
        help="Use tiles flagged as low-score (default: drop them from the blend).",
    )
    parser.add_argument(
        "--output-tile",
        type=int,
        default=1024,
        help="Output canvas tile size (pixels) for bounded-memory iteration.",
    )
    parser.add_argument(
        "--reference-channel",
        type=str,
        default=None,
        help="Override reference channel for the preview (default: manifest value).",
    )
    parser.add_argument(
        "--moving-channel",
        type=str,
        default=None,
        help="Override moving channel for resampling (default: manifest value).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "outputs" / "stitch_tiles"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip overlay/checkerboard preview PNGs.",
    )
    return parser.parse_args()


def _parse_channel(value: str | None) -> int | str | None:
    if value is None or str(value).lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _warp_stitched(
    mov_tensor: torch.Tensor,       # (B, C, mov_h, mov_w) loaded at ds_mov
    *,
    centers_ref: np.ndarray,        # (T, 2) tile centers in ref-full pixels (x, y)
    mats_mov_from_ref: np.ndarray,  # (T, 3, 3)
    out_h: int,
    out_w: int,
    ds_out: int,
    ds_mov: int,
    sigma: float,
    output_tile: int,
) -> torch.Tensor:
    """Resample ``mov_tensor`` through the blended piecewise-affine field.

    For each output pixel the moving sample coordinate is the
    Gaussian-distance-weighted blend of every tile's affine evaluated at
    that pixel's reference-full position. The output is produced one
    ``output_tile`` block at a time so the sampling grid is never
    materialized for the whole canvas (bounded memory).
    """
    device = mov_tensor.device
    B, C = int(mov_tensor.shape[0]), int(mov_tensor.shape[1])
    mov_h, mov_w = int(mov_tensor.shape[-2]), int(mov_tensor.shape[-1])
    centers = torch.as_tensor(centers_ref, dtype=torch.float32, device=device)  # (T, 2)
    mats = torch.as_tensor(mats_mov_from_ref, dtype=torch.float32, device=device)  # (T,3,3)
    inv_two_sigma_sq = 1.0 / (2.0 * float(sigma) ** 2)

    out = torch.zeros((B, C, out_h, out_w), dtype=torch.float32, device=device)

    for y0 in range(0, out_h, output_tile):
        y1 = min(out_h, y0 + output_tile)
        for x0 in range(0, out_w, output_tile):
            x1 = min(out_w, x0 + output_tile)
            ys = torch.arange(y0, y1, device=device, dtype=torch.float32)
            xs = torch.arange(x0, x1, device=device, dtype=torch.float32)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # (h, w)
            h, w = gy.shape
            # reference-full pixel coordinates of these output pixels
            rx = gx * ds_out
            ry = gy * ds_out
            ref_pts = torch.stack(
                [rx, ry, torch.ones_like(rx)], dim=-1
            ).reshape(-1, 3)  # (P, 3)

            # weights: (P, T) Gaussian on distance to each tile center
            dx = rx.reshape(-1, 1) - centers[:, 0].reshape(1, -1)
            dy = ry.reshape(-1, 1) - centers[:, 1].reshape(1, -1)
            d2 = dx * dx + dy * dy
            wts = torch.exp(-d2 * inv_two_sigma_sq)  # (P, T)
            wts = wts / wts.sum(dim=1, keepdim=True).clamp_min(1e-12)

            # each tile's affine applied to all points: (T, P, 3)
            mov_pts = torch.einsum("tij,pj->tpi", mats, ref_pts)  # (T, P, 3)
            mov_xy = mov_pts[..., :2]  # (T, P, 2), mov-full pixels
            blended = (wts.t().unsqueeze(-1) * mov_xy).sum(dim=0)  # (P, 2)

            # mov-full -> mov-tensor pixels (loaded at ds_mov), then normalize
            mx = blended[:, 0] / ds_mov
            my = blended[:, 1] / ds_mov
            x_norm = (2.0 * mx + 1.0) / float(mov_w) - 1.0
            y_norm = (2.0 * my + 1.0) / float(mov_h) - 1.0
            tile_grid = (
                torch.stack([x_norm, y_norm], dim=-1).reshape(1, h, w, 2).expand(B, -1, -1, -1)
            )
            out[..., y0:y1, x0:x1] = torch.nn.functional.grid_sample(
                mov_tensor,
                tile_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

    return out


def _two_color_overlay(reference: torch.Tensor, moving: torch.Tensor) -> np.ndarray:
    """Build an (H, W, 3) red/green overlay: reference=green, moving=red.

    Each image is reduced to a single channel, min-max normalized, and
    placed in its own RGB channel so overlap reads as yellow. This is far
    easier to read than the gray+red alpha blend at any resolution.
    """

    def _prep(t: torch.Tensor) -> torch.Tensor:
        x = t.detach().to(torch.float32)
        while x.dim() > 2:  # collapse batch/channel by taking channel-mean
            x = x.mean(dim=0)
        lo, hi = float(x.min()), float(x.max())
        if hi > lo:
            x = (x - lo) / (hi - lo)
        else:
            x = torch.zeros_like(x)
        return x

    ref = _prep(reference)
    mov = _prep(moving)
    h = min(ref.shape[0], mov.shape[0])
    w = min(ref.shape[1], mov.shape[1])
    ref, mov = ref[:h, :w], mov[:h, :w]
    rgb = torch.stack([mov, ref, torch.zeros_like(ref)], dim=-1)  # R=mov, G=ref
    return (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text())
    ref_source = SourceInfo.from_dict(manifest["reference"])
    mov_source = SourceInfo.from_dict(manifest["moving"])
    ref_path = Path(ref_source.source_path)
    mov_path = Path(mov_source.source_path)

    ref_channel = _parse_channel(args.reference_channel or manifest.get("reference_channel"))
    mov_channel = _parse_channel(args.moving_channel or manifest.get("moving_channel"))

    tiles = manifest["tiles"]
    if not args.include_low_score:
        tiles = [t for t in tiles if not t.get("flagged_low_score", False)]
    if not tiles:
        raise ValueError(
            "No usable tiles in manifest (all flagged low-score). "
            "Re-run with --include-low-score or adjust register_tiles --min-score."
        )

    centers = []
    mats_mov_from_ref = []
    for t in tiles:
        ry, rx, rh, rw = t["reference_region_yxhw"]
        centers.append([rx + rw / 2.0, ry + rh / 2.0])
        m_ref_from_mov = np.asarray(t["matrix_ref_full_from_mov_full"], dtype=np.float64)
        mats_mov_from_ref.append(np.linalg.inv(m_ref_from_mov))
    centers_arr = np.asarray(centers, dtype=np.float64)
    mats_arr = np.asarray(mats_mov_from_ref, dtype=np.float64)

    # tile-center spacing -> blend sigma
    if len(centers_arr) > 1:
        from scipy.spatial.distance import cdist  # local import; scipy is a dep

        d = cdist(centers_arr, centers_arr)
        np.fill_diagonal(d, np.inf)
        spacing = float(np.median(d.min(axis=1)))
    else:
        ref_h, ref_w = ref_source.base_shape_hw
        spacing = float(max(ref_h, ref_w))
    sigma = max(args.blend_sigma * spacing, 1.0)

    ref_h, ref_w = ref_source.base_shape_hw
    out_h = max(1, ref_h // args.output_downsample)
    out_w = max(1, ref_w // args.output_downsample)

    print(f"reference: {ref_path}  full={ref_source.base_shape_hw}")
    print(f"moving:    {mov_path}  full={mov_source.base_shape_hw}")
    print(f"usable tiles: {len(tiles)}  center spacing={spacing:.1f}px  sigma={sigma:.1f}px")
    print(f"output canvas: {out_h}x{out_w} (ref/{args.output_downsample})")

    moving = load_image(
        mov_path, channels=mov_channel, downsample=args.moving_downsample, grayscale=False
    )
    mov_tensor = moving.detail.image.to(torch.float32)

    warped = _warp_stitched(
        mov_tensor,
        centers_ref=centers_arr,
        mats_mov_from_ref=mats_arr,
        out_h=out_h,
        out_w=out_w,
        ds_out=args.output_downsample,
        ds_mov=args.moving_downsample,
        sigma=sigma,
        output_tile=args.output_tile,
    )

    out_path = args.output_dir / "stitched_registered.ome.tif"
    out_source = SourceInfo(
        source_path=str(out_path),
        series_index=0,
        level_count=1,
        level_shapes=((out_h, out_w),),
        axes=mov_source.axes,
        dtype=mov_source.dtype,
        channel_names=mov_source.channel_names,
        pixel_size_xy=(
            tuple(p * args.output_downsample for p in ref_source.pixel_size_xy)
            if ref_source.pixel_size_xy
            else None
        ),
        unit=ref_source.unit,
        origin_xy=ref_source.origin_xy,
        format="ome-tiff",
    )
    save_tiff(
        warped,
        out_path,
        config=TiffExportConfig(format="ome-tiff", photometric="auto", overwrite=True),
        source=out_source,
    )
    print(f"stitched warped moving: {out_path}")

    if not args.no_preview:
        import matplotlib.image as mpimg

        reference = load_image(
            ref_path, channels=ref_channel, downsample=args.output_downsample, grayscale=False
        )
        ref_detail = reference.detail
        # match canvas (loader may differ by a pixel from ref_h//ds)
        rh = min(ref_detail.image.shape[-2], warped.shape[-2])
        rw = min(ref_detail.image.shape[-1], warped.shape[-1])
        ref_crop = ref_detail.image[..., :rh, :rw]
        warped_crop = warped[..., :rh, :rw]
        from batchmatch.base.detail import build_image_td

        # Two-color overlay: reference=green, moving=red, overlap=yellow.
        two_color = _two_color_overlay(ref_crop, warped_crop)
        two_color_path = args.output_dir / "preview_two_color.png"
        mpimg.imsave(str(two_color_path), two_color)

        show_comparison(
            build_image_td(ref_crop),
            build_image_td(warped_crop),
            mode="overlay",
            spec=OverlaySpec(alpha=0.5),
            display=DisplaySpec(
                title="Stitched Registered Overlay",
                show=False,
                save_path=str(args.output_dir / "preview_overlay.png"),
            ),
        )
        show_comparison(
            build_image_td(ref_crop),
            build_image_td(warped_crop),
            mode="checkerboard",
            spec=CheckerboardSpec(
                tiles=(24, 24),
                edge_overlay=EdgeOverlaySpec(edge_source="mov", edge_threshold=0.3),
            ),
            display=DisplaySpec(
                title="Stitched Registered Checkerboard",
                show=False,
                save_path=str(args.output_dir / "preview_checkerboard.png"),
            ),
        )
        print(f"previews: {two_color_path} (two-color), {args.output_dir / 'preview_overlay.png'}")

    field_manifest = {
        "manifest": str(args.manifest.resolve()),
        "output_downsample": args.output_downsample,
        "moving_downsample": args.moving_downsample,
        "blend_sigma_px": sigma,
        "tile_center_spacing_px": spacing,
        "usable_tiles": len(tiles),
        "output_shape_hw": [out_h, out_w],
    }
    (args.output_dir / "stitch_info.json").write_text(json.dumps(field_manifest, indent=2))


if __name__ == "__main__":
    main()
