"""Register two images of cells via exhaustive warp search.

- Pre-scales images based on estimated cell size (``CellUnitResize``).
- Runs an exhaustive warp search on a common low-res search canvas.
- Saves a :class:`RegistrationTransform` manifest + optional previews and
  exports a full-resolution aligned TIFF with reference and registered
  moving image stored as separate prefixed channels.

Run:
    uv run examples/registration/register_cells.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from batchmatch import auto_device
from batchmatch.gradient import (
    CDGradientConfig,
    EtaConfig,
    L2NormConfig,
    NormalizeConfig,
)
from batchmatch.io import (
    ProductIO,
    export_stacked_registered_tiff as export_stacked_registered_tiff_core,
    load_image,
)
from batchmatch.io.space import SpatialImage
from batchmatch.process.cells import CellSizeCfg
from batchmatch.process.pad import CenterPad
from batchmatch.process.resize import CellUnitResize, TargetResize
from batchmatch.process.spatial_stages import (
    SpatialCenterPad,
    SpatialTargetResize,
    SpatialUnitResize,
)
from batchmatch.search import (
    ExhaustiveSearchConfig,
    ExhaustiveWarpSearch,
    SearchParams,
)
from batchmatch.search.config import ScaleRange
from batchmatch.search.transform import RegistrationTransform
from batchmatch.translate.config import (
    GPCTranslationConfig,
    NCCTranslationConfig,
    NGFTranslationConfig,
)
from batchmatch.view.config import CheckerboardSpec, EdgeOverlaySpec, OverlaySpec
from batchmatch.view.composite import render_checkerboard, render_overlay
from batchmatch.view.display import show_comparison
from batchmatch.view.preview import render_registration_preview


def _parse_image_from_selection(selection: int) -> tuple[Path, Path]:
    base_path = Path("cells") / f"selection_{selection}"
    moving_path = base_path / "fish_dapi.jpg"
    reference_path = base_path / f"xenium_region_2-sel{selection}-highres_wide.png"
    return moving_path, reference_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=int, default=7)
    parser.add_argument(
        "--moving",
        type=Path,
        default=None,
        help="Moving image path (e.g. FISH jpg). Overrides --selection when given.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Reference image path (e.g. a Xenium tile png). Overrides --selection when given.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "register_cells")
    parser.add_argument("--search-dim", type=int, default=1024)
    parser.add_argument("--metric", choices=("ncc", "ngf", "gpc"), default="gpc")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-prefix", type=str, default="reg_cells")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-export", action="store_true", help="Skip full-res export.")
    parser.add_argument("--export-name", type=str, default="registered_fullres.ome.tif")
    parser.add_argument(
        "--export-canvas",
        choices=("union", "reference"),
        default="union",
        help="Canvas for the full-resolution stacked export.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help=(
            "Optional output tile size for full-resolution TIFF export. "
            "Streaming export defaults to 512 and rounds to a multiple of 16."
        ),
    )
    parser.add_argument(
        "--pyramid-levels",
        type=int,
        default=4,
        help="Number of 2x OME-TIFF SubIFD pyramid levels for exported TIFFs.",
    )
    parser.add_argument(
        "--eager-export",
        action="store_true",
        help="Disable streamed tile/channel TIFF export and use the legacy eager path.",
    )
    parser.add_argument(
        "--export-compression",
        type=str,
        default=None,
        help="Optional lossless tifffile compression codec, e.g. zlib or zstd.",
    )
    return parser.parse_args()


def _str_to_translation_config(metric: str):
    if metric == "ncc":
        return NCCTranslationConfig()
    if metric == "ngf":
        return NGFTranslationConfig()
    if metric == "gpc":
        return GPCTranslationConfig()
    raise ValueError(f"Unsupported metric: {metric}")


def export_stacked_registered_tiff(
    transform: RegistrationTransform,
    *,
    moving: SpatialImage,
    reference: SpatialImage,
    output_path: Path,
    canvas: str = "union",
    tile_size: int | None = None,
    pyramid_levels: int = 4,
    streaming: bool = True,
    compression: str | None = None,
    overwrite: bool = True,
) -> Path:
    """Save reference and registered moving image as prefixed TIFF channels."""
    export_stacked_registered_tiff_core(
        transform,
        output_path=output_path,
        reference_path=reference.space.source.path,
        moving_path=moving.space.source.path,
        canvas=canvas,  # type: ignore[arg-type]
        tile_size=tile_size,
        pyramid_levels=pyramid_levels,
        streaming=streaming,
        overwrite=overwrite,
        compression=compression,
    )
    return output_path


def main() -> None:
    args = _parse_args()
    if args.moving is not None or args.reference is not None:
        if args.moving is None or args.reference is None:
            raise ValueError("Provide both --moving and --reference, or neither.")
        moving_path, reference_path = args.moving, args.reference
    else:
        moving_path, reference_path = _parse_image_from_selection(args.selection)
    print(f"moving:    {moving_path}")
    print(f"reference: {reference_path}")

    moving = load_image(moving_path, grayscale=True)
    reference = load_image(reference_path, grayscale=True)

    cell_cfg = CellSizeCfg(
        radius_hint_px=14.0,
        use_log=True,
        blob_weight=0.4,
        max_dim=1024,
        morph_radius_px=3,
        peak_rel_threshold=0.3,
    )

    unit = SpatialUnitResize(inner=CellUnitResize(cell_cfg=cell_cfg))
    resize = SpatialTargetResize(inner=TargetResize(target_width=args.search_dim))
    pad = SpatialCenterPad(
        inner=CenterPad(
            scale=2,
            window_alpha=0.05,
            pad_to_pow2=False,
            outputs=["image", "box", "mask", "quad", "window"],
        )
    )
    prepare = unit >> resize >> pad

    # clone: inner stages may mutate their input details; keep originals
    # intact so we can still use them for preview/export.
    moving_search, reference_search = prepare([moving.clone(), reference.clone()])

    print(f"search image shape: {tuple(reference_search.detail.image.shape)}")
    print(f"device: {args.device}")

    search_params = SearchParams(
        scale_x=ScaleRange(min_scale=0.9, max_scale=1.5, step=0.01),
        scale_y=ScaleRange(min_scale=0.9, max_scale=1.5, step=0.01),
    )
    gradient = (
        CDGradientConfig(
            eta=EtaConfig.from_mean(scale=0.2, norm=L2NormConfig()),
            normalize=NormalizeConfig(norm="l2", threshold=1e-3),
        )
        if args.metric == "ngf"
        else CDGradientConfig()
    )
    config = ExhaustiveSearchConfig(
        translation=_str_to_translation_config(args.metric),
        batch_size=4,
        progress_enabled=True,
        gradient=gradient,
    )
    search = ExhaustiveWarpSearch(search_params, config)

    device = auto_device(args.device)
    reference_search = reference_search.to(device)
    moving_search = moving_search.to(device)
    search = search.to(device)

    start_t = time.perf_counter()
    result = search(reference_search.detail, moving_search.detail, top_k=1, progress=True)
    print(f"Search took {time.perf_counter() - start_t:.2f} s.")

    warp = result.warp
    print(
        f"warp: angle={warp.angle.item():.2f}, "
        f"scale_x={warp.scale_x.item():.4f}, scale_y={warp.scale_y.item():.4f}"
    )

    moving_search_cpu = moving_search.to("cpu")
    reference_search_cpu = reference_search.to("cpu")
    result_cpu = result.to("cpu")

    transform = RegistrationTransform.from_search(
        moving=moving_search_cpu,
        reference=reference_search_cpu,
        search_result=result_cpu,
    )

    preview = render_registration_preview(
        transform,
        moving.to("cpu"),
        reference.to("cpu"),
        crop_mode="union",
    )

    checkerboard_spec = CheckerboardSpec(
        tiles=(64, 64),
        edge_overlay=EdgeOverlaySpec(edge_source="mov", edge_threshold=0.3),
    )
    overlay_spec = OverlaySpec(alpha=0.5)
    checkerboard = render_checkerboard(preview.reference, preview.moving_warped, checkerboard_spec)
    overlay = render_overlay(preview.reference, preview.moving_warped, overlay_spec)

    if args.show:
        show_comparison(preview.reference, preview.moving_warped, mode="overlay", spec=overlay_spec)

    out_dir = args.output_dir / args.output_prefix / f"metric_{args.metric}_searchdim_{args.search_dim}"
    store = ProductIO(out_dir)
    manifest_path = store.save(
        transform,
        preview=preview,
        checkerboard=checkerboard,
        overlay=overlay,
        debug_details={"search_result": result_cpu},
        overwrite=True,
    )
    print(f"manifest: {manifest_path}")

    if not args.no_export:
        export_path = export_stacked_registered_tiff(
            transform,
            moving=moving.to("cpu"),
            reference=reference.to("cpu"),
            output_path=out_dir / args.export_name,
            canvas=args.export_canvas,
            tile_size=args.tile_size,
            pyramid_levels=args.pyramid_levels,
            streaming=not args.eager_export,
            compression=args.export_compression,
            overwrite=True,
        )
        print(f"exported: {export_path}")


if __name__ == "__main__":
    main()
