"""Tile-wise low-resolution registration of two calibrated TIFFs.

Co-tiles a reference and moving image by *physical* world coordinates
(using each TIFF's ``origin_xy`` and ``pixel_size_xy``) so that
corresponding tiles cover the same region of space, then runs the
exhaustive warp search independently on each tile pair at a reduced
resolution.

This is the "register tiles at lower resolution" step that whole-image
registration cannot resolve on wide-view Xenium vs. FISH data: each tile
gets its own local affine transform, expressed directly in
full-resolution source pixel coordinates.

Run:
    uv run python examples/registration/register_tiles.py
    uv run python examples/registration/register_tiles.py \
        --reference ref.ome.tif --moving mov.ome.tif \
        --tile-size 2000 2000 --downsample 4
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from batchmatch import auto_device
from batchmatch.io import (
    RegionYXHW,
    SourceInfo,
    load_image,
    open_image,
)
from batchmatch.io.space import SpatialImage
from batchmatch.process.pad import CenterPad
from batchmatch.process.resize import TargetResize
from batchmatch.process.spatial_stages import (
    SpatialCenterPad,
    SpatialPhysicalResize,
    SpatialTargetResize,
)
from batchmatch.search import (
    AngleRange,
    ExhaustiveSearchConfig,
    ExhaustiveWarpSearch,
    ScaleRange,
    SearchParams,
    ShearRange,
)
from batchmatch.search.transform import RegistrationTransform
from batchmatch.view.config import (
    CheckerboardSpec,
    DisplaySpec,
    EdgeOverlaySpec,
)
from batchmatch.view.display import show_comparison
from batchmatch.warp.resample import warp_to_reference


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reference", type=Path, default=root / "img" / "FISH.tif")
    parser.add_argument("--moving", type=Path, default=None)
    parser.add_argument("--reference-channel", type=str, default="DAPI")
    parser.add_argument("--moving-channel", type=str, default=None)
    parser.add_argument(
        "--tile-size",
        type=float,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help=(
            "Tile size in physical units matching pixel_size_xy (e.g. microns). "
            "Defaults to a 2x2 grid over the shared physical region."
        ),
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Overlap between adjacent tiles, in the same physical units as --tile-size.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help=(
            "Integer downsample applied when reading each tile. Default 1 "
            "(full resolution per tile). Set >1 for a faster low-resolution search."
        ),
    )
    parser.add_argument("--search-dim", type=int, default=512)
    parser.add_argument("--pad-scale", type=float, default=2.0)
    parser.add_argument(
        "--metric",
        choices=("ncc", "pc", "cc", "gcc", "gpc", "ngf", "gngf"),
        default="ncc",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--rotation", type=float, nargs=3, metavar=("MIN", "MAX", "STEP"), default=(-3.0, 3.0, 1.0)
    )
    parser.add_argument(
        "--scale-x", type=float, nargs=3, metavar=("MIN", "MAX", "STEP"), default=(0.95, 1.05, 0.025)
    )
    parser.add_argument(
        "--scale-y", type=float, nargs=3, metavar=("MIN", "MAX", "STEP"), default=(0.95, 1.05, 0.025)
    )
    parser.add_argument(
        "--shear-x", type=float, nargs=3, metavar=("MIN", "MAX", "STEP"), default=(0.0, 0.0, 1.0)
    )
    parser.add_argument(
        "--shear-y", type=float, nargs=3, metavar=("MIN", "MAX", "STEP"), default=(0.0, 0.0, 1.0)
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Flag tiles whose best translation score falls below this threshold.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "outputs" / "register_tiles"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Save a low-res registered checkerboard PNG per tile.",
    )
    return parser.parse_args()


def _parse_channel(value: str | None) -> int | str | None:
    if value is None or value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _range3(values: Sequence[float], name: str) -> tuple[float, float, float]:
    start, end, step = (float(v) for v in values)
    if step <= 0:
        raise ValueError(f"{name} step must be positive.")
    return start, end, step


def _build_search_params(args: argparse.Namespace) -> SearchParams:
    r0, r1, rs = _range3(args.rotation, "--rotation")
    sx0, sx1, sxs = _range3(args.scale_x, "--scale-x")
    sy0, sy1, sys = _range3(args.scale_y, "--scale-y")
    hx0, hx1, hxs = _range3(args.shear_x, "--shear-x")
    hy0, hy1, hys = _range3(args.shear_y, "--shear-y")
    return SearchParams(
        rotation=AngleRange(r0, r1, rs),
        scale_x=ScaleRange(sx0, sx1, sxs),
        scale_y=ScaleRange(sy0, sy1, sys),
        shear_x=ShearRange(hx0, hx1, hxs),
        shear_y=ShearRange(hy0, hy1, hys),
    )


def _world_bbox(source: SourceInfo) -> tuple[float, float, float, float]:
    """Physical bounding box (x0, y0, x1, y1) of a calibrated source."""
    if source.pixel_size_xy is None:
        raise ValueError(
            f"{source.source_path} lacks pixel_size_xy; physical co-tiling requires "
            "spatial calibration. Re-tile with --tile-size-pixels or supply OME-TIFF metadata."
        )
    px_w, px_h = source.pixel_size_xy
    ox, oy = source.origin_xy or (0.0, 0.0)
    h, w = source.base_shape_hw
    return (ox, oy, ox + w * px_w, oy + h * px_h)


def _world_to_region(
    source: SourceInfo,
    wx0: float,
    wy0: float,
    wx1: float,
    wy1: float,
) -> RegionYXHW | None:
    """Convert a physical box to a pixel RegionYXHW clipped to the image."""
    px_w, px_h = source.pixel_size_xy  # type: ignore[misc]
    ox, oy = source.origin_xy or (0.0, 0.0)
    h, w = source.base_shape_hw

    x_px = (wx0 - ox) / px_w
    y_px = (wy0 - oy) / px_h
    x2_px = (wx1 - ox) / px_w
    y2_px = (wy1 - oy) / px_h

    x = max(0, int(np.floor(min(x_px, x2_px))))
    y = max(0, int(np.floor(min(y_px, y2_px))))
    x2 = min(w, int(np.ceil(max(x_px, x2_px))))
    y2 = min(h, int(np.ceil(max(y_px, y2_px))))
    if x2 - x < 1 or y2 - y < 1:
        return None
    return RegionYXHW(y=y, x=x, h=y2 - y, w=x2 - x)


def _shared_tile_grid(
    ref: SourceInfo,
    mov: SourceInfo,
    tile_h: float | None,
    tile_w: float | None,
    overlap: float,
) -> list[tuple[float, float, float, float]]:
    """Tile the intersection of the two physical bounding boxes."""
    rx0, ry0, rx1, ry1 = _world_bbox(ref)
    mx0, my0, mx1, my1 = _world_bbox(mov)
    x0, y0 = max(rx0, mx0), max(ry0, my0)
    x1, y1 = min(rx1, mx1), min(ry1, my1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            "Reference and moving images do not overlap in physical space. "
            f"ref bbox={(rx0, ry0, rx1, ry1)}, mov bbox={(mx0, my0, mx1, my1)}."
        )

    if tile_h is None or tile_w is None:
        tile_h = (y1 - y0) / 2.0
        tile_w = (x1 - x0) / 2.0

    step_w = max(tile_w - overlap, tile_w * 1e-3)
    step_h = max(tile_h - overlap, tile_h * 1e-3)

    tiles: list[tuple[float, float, float, float]] = []
    wy = y0
    while wy < y1:
        wx = x0
        while wx < x1:
            tiles.append((wx, wy, min(wx + tile_w, x1), min(wy + tile_h, y1)))
            wx += step_w
        wy += step_h
    return tiles


def _register_pair(
    reference: SpatialImage,
    moving: SpatialImage,
    *,
    search_params: SearchParams,
    config: ExhaustiveSearchConfig,
    search_dim: int,
    pad_scale: float,
    device: torch.device,
) -> tuple[RegistrationTransform, SpatialImage, SpatialImage, object]:
    prepare = (
        SpatialPhysicalResize(reference_index=0)
        >> SpatialTargetResize(inner=TargetResize(target_width=search_dim))
        >> SpatialCenterPad(
            inner=CenterPad(
                scale=pad_scale,
                window_alpha=0.05,
                pad_to_pow2=False,
                outputs=["image", "box", "mask", "quad", "window"],
            )
        )
    )
    ref_search, mov_search = prepare([reference.clone(), moving.clone()])

    search = ExhaustiveWarpSearch(search_params, config).to(device)
    result = search.search(
        ref_search.to(device).detail,
        mov_search.to(device).detail,
        top_k=1,
        progress=False,
    )
    result_cpu = result.to("cpu")
    ref_search_cpu = ref_search.to("cpu")
    mov_search_cpu = mov_search.to("cpu")
    transform = RegistrationTransform.from_search(
        moving=mov_search_cpu,
        reference=ref_search_cpu,
        search_result=result_cpu,
    )
    return transform, ref_search_cpu, mov_search_cpu, result_cpu


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = auto_device(args.device)
    moving_path = args.moving or args.reference
    ref_channel = _parse_channel(args.reference_channel)
    mov_channel = (
        _parse_channel(args.moving_channel) if args.moving_channel is not None else ref_channel
    )

    with open_image(args.reference) as src:
        ref_source = src.source
    with open_image(moving_path) as src:
        mov_source = src.source

    if (ref_source.unit or "") != (mov_source.unit or ""):
        raise ValueError(
            f"Physical units differ: ref={ref_source.unit!r}, mov={mov_source.unit!r}."
        )

    tile_h = args.tile_size[0] if args.tile_size else None
    tile_w = args.tile_size[1] if args.tile_size else None
    grid = _shared_tile_grid(ref_source, mov_source, tile_h, tile_w, args.overlap)

    unit = ref_source.unit or "units"
    print(f"reference: {args.reference}  shape={ref_source.base_shape_hw}")
    print(f"moving:    {moving_path}  shape={mov_source.base_shape_hw}")
    print(f"shared tiles: {len(grid)}  downsample={args.downsample}  unit={unit}")

    search_params = _build_search_params(args)
    config = ExhaustiveSearchConfig(
        batch_size=args.batch_size,
        translation_method=args.metric,
        progress_enabled=False,
        device=device,
    )

    tiles_out: list[dict] = []
    skipped = 0
    start_t = time.perf_counter()

    for idx, (wx0, wy0, wx1, wy1) in enumerate(grid):
        ref_region = _world_to_region(ref_source, wx0, wy0, wx1, wy1)
        mov_region = _world_to_region(mov_source, wx0, wy0, wx1, wy1)
        if ref_region is None or mov_region is None:
            skipped += 1
            continue

        reference = load_image(
            args.reference,
            channels=ref_channel,
            region=ref_region,
            downsample=args.downsample,
            grayscale=False,
        )
        moving = load_image(
            moving_path,
            channels=mov_channel,
            region=mov_region,
            downsample=args.downsample,
            grayscale=False,
        )

        transform, ref_search, mov_search, result = _register_pair(
            reference,
            moving,
            search_params=search_params,
            config=config,
            search_dim=args.search_dim,
            pad_scale=args.pad_scale,
            device=device,
        )

        warp = result.warp
        tr = result.translation_results
        score = float(tr.score[0].item())
        flagged = args.min_score is not None and score < args.min_score

        if args.preview:
            low_registered = warp_to_reference(
                mov_search.detail,
                transform.matrix_ref_search_from_mov_search,
                out_hw=ref_search.shape_hw,
            )
            show_comparison(
                ref_search.detail,
                low_registered,
                mode="checkerboard",
                spec=CheckerboardSpec(
                    tiles=(16, 16),
                    edge_overlay=EdgeOverlaySpec(edge_source="mov", edge_threshold=0.3),
                ),
                display=DisplaySpec(
                    title=f"tile {idx:04d} (score={score:.3f})",
                    show=False,
                    save_path=str(args.output_dir / f"tile_{idx:04d}_checkerboard.png"),
                ),
            )

        entry = {
            "index": idx,
            "world_bbox_xyxy": [wx0, wy0, wx1, wy1],
            "reference_region_yxhw": ref_region.to_list(),
            "moving_region_yxhw": mov_region.to_list(),
            "warp": {
                "angle": float(warp.angle[0].item()),
                "scale_x": float(warp.scale_x[0].item()),
                "scale_y": float(warp.scale_y[0].item()),
                "shear_x": float(warp.shear_x[0].item()),
                "shear_y": float(warp.shear_y[0].item()),
                "tx": float(tr.tx[0].item()),
                "ty": float(tr.ty[0].item()),
            },
            "score": score,
            "flagged_low_score": flagged,
            "matrix_ref_full_from_mov_full": transform.matrix_ref_full_from_mov_full.tolist(),
            "matrix_ref_phys_from_mov_phys": (
                transform.matrix_ref_phys_from_mov_phys.tolist()
                if transform.matrix_ref_phys_from_mov_phys is not None
                else None
            ),
        }
        tiles_out.append(entry)
        flag = "  [LOW SCORE]" if flagged else ""
        print(
            f"  [{idx + 1}/{len(grid)}] score={score:.4f} "
            f"angle={entry['warp']['angle']:.2f} "
            f"sx={entry['warp']['scale_x']:.4f} sy={entry['warp']['scale_y']:.4f} "
            f"tx={entry['warp']['tx']:.1f} ty={entry['warp']['ty']:.1f}{flag}"
        )

    elapsed = time.perf_counter() - start_t

    manifest = {
        "reference": ref_source.to_dict(),
        "moving": mov_source.to_dict(),
        "reference_channel": str(ref_channel),
        "moving_channel": str(mov_channel),
        "tiling": {
            "tile_size_phys_hw": [tile_h, tile_w] if args.tile_size else None,
            "overlap_phys": args.overlap,
            "downsample": args.downsample,
            "unit": unit,
            "n_tiles": len(tiles_out),
            "n_skipped": skipped,
        },
        "search": {
            "search_dim": args.search_dim,
            "pad_scale": args.pad_scale,
            "metric": args.metric,
            "rotation": list(args.rotation),
            "scale_x": list(args.scale_x),
            "scale_y": list(args.scale_y),
            "shear_x": list(args.shear_x),
            "shear_y": list(args.shear_y),
        },
        "tiles": tiles_out,
    }
    manifest_path = args.output_dir / "tile_transforms.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    n_flagged = sum(1 for t in tiles_out if t["flagged_low_score"])
    print(f"\nregistered {len(tiles_out)} tiles in {elapsed:.1f}s ({skipped} skipped)")
    if args.min_score is not None:
        print(f"low-score tiles (< {args.min_score}): {n_flagged}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
