"""Tile-wise registration of two calibrated TIFFs via coarse-then-tile.

A coarse whole-image alignment is computed first (or loaded from a prior
register_tiff manifest with ``--init-manifest``). The reference image is
then tiled in its own pixel space, and for each reference tile the coarse
transform predicts the corresponding moving region, which is loaded and
refined with an independent exhaustive warp search. Each tile yields a
local affine transform expressed directly in full-resolution source
coordinates.

This coarse-then-tile correspondence is required for cross-instrument
data (e.g. Olympus FISH vs. Xenium morphology): the two images do NOT
share a world coordinate frame, so tiling by absolute physical
coordinates would pair unrelated regions. The coarse alignment puts them
into correspondence first.

For multimodal pairs use a gradient-based metric (``--metric ngf`` or
``gngf``); intensity NCC is unreliable across modalities.

Run:
    uv run python examples/registration/register_tiles.py \
        --reference ref.ome.tif --moving mov.ome.tif \
        --reference-channel DAPI --moving-channel 0 \
        --tile-size 2000 2000 --metric ngf
    # reuse an existing coarse alignment:
    uv run python examples/registration/register_tiles.py ... \
        --init-manifest outputs/register_tiff/registration.json
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
    parser.add_argument(
        "--coarse-downsample",
        type=int,
        default=16,
        help=(
            "Downsample for the coarse whole-image alignment that establishes "
            "tile correspondence. Larger = faster/coarser. Ignored if "
            "--init-manifest is given."
        ),
    )
    parser.add_argument(
        "--init-manifest",
        type=Path,
        default=None,
        help=(
            "Path to a register_tiff registration.json. Its "
            "ref_full_from_mov_full matrix is used as the coarse alignment "
            "instead of computing one internally."
        ),
    )
    parser.add_argument(
        "--moving-margin",
        type=float,
        default=0.5,
        help=(
            "Fraction of the tile size by which each tile's predicted moving "
            "region is expanded on every side, to absorb coarse-alignment error."
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


def _reference_tile_grid(
    ref: SourceInfo,
    tile_h_phys: float | None,
    tile_w_phys: float | None,
    overlap_phys: float,
) -> list[RegionYXHW]:
    """Tile the reference image in its own pixel space.

    Tile sizes are given in physical units (matching ``ref.pixel_size_xy``)
    and converted to reference pixels. When no tile size is supplied, the
    reference is split into a 2x2 grid.
    """
    ref_h, ref_w = ref.base_shape_hw
    if tile_h_phys is None or tile_w_phys is None or ref.pixel_size_xy is None:
        tile_h = max(1, ref_h // 2)
        tile_w = max(1, ref_w // 2)
        overlap_h = overlap_w = 0
    else:
        px_w, px_h = ref.pixel_size_xy
        tile_h = max(1, round(tile_h_phys / px_h))
        tile_w = max(1, round(tile_w_phys / px_w))
        overlap_h = max(0, round(overlap_phys / px_h))
        overlap_w = max(0, round(overlap_phys / px_w))

    step_h = max(1, tile_h - overlap_h)
    step_w = max(1, tile_w - overlap_w)

    tiles: list[RegionYXHW] = []
    y = 0
    while y < ref_h:
        h = min(tile_h, ref_h - y)
        x = 0
        while x < ref_w:
            w = min(tile_w, ref_w - x)
            tiles.append(RegionYXHW(y=y, x=x, h=h, w=w))
            x += step_w
        y += step_h
    return tiles


def _apply_affine(pts_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3x3 affine to (N, 2) points in (x, y)."""
    homog = np.concatenate([pts_xy, np.ones((pts_xy.shape[0], 1))], axis=1)
    out = homog @ np.asarray(matrix, dtype=np.float64).T
    return out[:, :2]


def _predict_moving_region(
    ref_region: RegionYXHW,
    matrix_mov_from_ref: np.ndarray,
    mov: SourceInfo,
    margin: float,
) -> RegionYXHW | None:
    """Map a reference tile through the coarse transform to a moving region.

    The reference tile's four corners are mapped to moving-full pixel
    coordinates; their bounding box (expanded by ``margin`` of the tile
    size on each side) is clipped to the moving image.
    """
    mov_h, mov_w = mov.base_shape_hw
    corners = np.array(
        [
            [ref_region.x, ref_region.y],
            [ref_region.x2, ref_region.y],
            [ref_region.x, ref_region.y2],
            [ref_region.x2, ref_region.y2],
        ],
        dtype=np.float64,
    )
    mov_pts = _apply_affine(corners, matrix_mov_from_ref)
    mx0, my0 = mov_pts[:, 0].min(), mov_pts[:, 1].min()
    mx1, my1 = mov_pts[:, 0].max(), mov_pts[:, 1].max()
    pad_x = (mx1 - mx0) * margin
    pad_y = (my1 - my0) * margin

    x = max(0, int(np.floor(mx0 - pad_x)))
    y = max(0, int(np.floor(my0 - pad_y)))
    x2 = min(mov_w, int(np.ceil(mx1 + pad_x)))
    y2 = min(mov_h, int(np.ceil(my1 + pad_y)))
    if x2 - x < 1 or y2 - y < 1:
        return None
    return RegionYXHW(y=y, x=x, h=y2 - y, w=x2 - x)


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


def _coarse_matrix_from_manifest(path: Path) -> np.ndarray:
    """Read ref_full_from_mov_full from a register_tiff registration.json."""
    data = json.loads(path.read_text())
    try:
        mat = data["transform"]["matrices"]["ref_full_from_mov_full"]
    except (KeyError, TypeError) as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"{path} does not look like a register_tiff manifest "
            "(missing transform.matrices.ref_full_from_mov_full)."
        ) from exc
    return np.asarray(mat, dtype=np.float64)


def _compute_coarse_matrix(
    reference_path: Path,
    moving_path: Path,
    *,
    ref_channel: int | str | None,
    mov_channel: int | str | None,
    coarse_downsample: int,
    search_params: SearchParams,
    config: ExhaustiveSearchConfig,
    search_dim: int,
    pad_scale: float,
    device: torch.device,
) -> np.ndarray:
    """Run a whole-image low-res registration to get ref_full_from_mov_full."""
    reference = load_image(
        reference_path, channels=ref_channel, downsample=coarse_downsample, grayscale=False
    )
    moving = load_image(
        moving_path, channels=mov_channel, downsample=coarse_downsample, grayscale=False
    )
    transform, _, _, result = _register_pair(
        reference,
        moving,
        search_params=search_params,
        config=config,
        search_dim=search_dim,
        pad_scale=pad_scale,
        device=device,
    )
    score = float(result.translation_results.score[0].item())
    print(f"coarse alignment score={score:.4f} (downsample={coarse_downsample})")
    return transform.matrix_ref_full_from_mov_full


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

    unit = ref_source.unit or "units"
    print(f"reference: {args.reference}  shape={ref_source.base_shape_hw}")
    print(f"moving:    {moving_path}  shape={mov_source.base_shape_hw}")

    search_params = _build_search_params(args)
    config = ExhaustiveSearchConfig(
        batch_size=args.batch_size,
        translation_method=args.metric,
        progress_enabled=False,
        device=device,
    )

    # --- Coarse global alignment defines tile correspondence -------------
    if args.init_manifest is not None:
        coarse_ref_from_mov = _coarse_matrix_from_manifest(args.init_manifest)
        print(f"coarse alignment: loaded from {args.init_manifest}")
    else:
        coarse_ref_from_mov = _compute_coarse_matrix(
            args.reference,
            moving_path,
            ref_channel=ref_channel,
            mov_channel=mov_channel,
            coarse_downsample=args.coarse_downsample,
            search_params=search_params,
            config=config,
            search_dim=args.search_dim,
            pad_scale=args.pad_scale,
            device=device,
        )
    coarse_mov_from_ref = np.linalg.inv(coarse_ref_from_mov)

    tile_h = args.tile_size[0] if args.tile_size else None
    tile_w = args.tile_size[1] if args.tile_size else None
    grid = _reference_tile_grid(ref_source, tile_h, tile_w, args.overlap)
    print(f"reference tiles: {len(grid)}  downsample={args.downsample}  unit={unit}")

    tiles_out: list[dict] = []
    skipped = 0
    start_t = time.perf_counter()

    for idx, ref_region in enumerate(grid):
        mov_region = _predict_moving_region(
            ref_region, coarse_mov_from_ref, mov_source, args.moving_margin
        )
        if mov_region is None:
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
            "moving_margin": args.moving_margin,
        },
        "coarse": {
            "source": (
                str(args.init_manifest) if args.init_manifest is not None else "internal"
            ),
            "downsample": (
                None if args.init_manifest is not None else args.coarse_downsample
            ),
            "matrix_ref_full_from_mov_full": coarse_ref_from_mov.tolist(),
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
