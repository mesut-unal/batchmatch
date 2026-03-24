"""
Example of registering two images of cells using exhaustive warp search.
- Prescales images based on cell size estimates to reduce search space.
- Saves registered images and search results to output directory.;lkjhg

Run:
    uv run examples/registration/register_cells.py
    uv run python examples/registration/register_cells.py --selection 13
"""


import time
from pathlib import Path
import torch
import torch.nn.functional as F

from batchmatch.base import ImageDetail, build_image_td
from batchmatch.io import ImageIO, ProductIO
from batchmatch.process.cells import CellSizeCfg
from batchmatch.process.crop import RandomCropStage
from batchmatch.process.pad import CenterPad
from batchmatch.process.resize import ScaleResize, TargetResize, CellUnitResize
from batchmatch.translate import NGFTranslationConfig
from batchmatch.gradient import (
    CDGradientConfig,
    EtaConfig,
    L2NormConfig,
    NormalizeConfig,
)
from batchmatch.search import (
    ExhaustiveSearchConfig,
    ExhaustiveWarpSearch,
    AngleRange,
    SearchParams,
    build_product_pipeline,
)
from batchmatch.search.config import ScaleRange
from batchmatch.translate.config import GNGFTranslationConfig, GPCTranslationConfig, NCCTranslationConfig
from batchmatch.view.config import CheckerboardSpec, EdgeOverlaySpec, OverlaySpec
from batchmatch.view.display import show_comparison
from batchmatch.warp import WarpPipelineConfig, build_warp_pipeline

import argparse

def _parse_image_from_selection(selection: int, tile: str, xenIN: str) -> tuple[Path, Path]:
    base_path = Path("cells") / f"selection_{selection}"
    moving_path = base_path / "fish_dapi.jpg"
    sel = f"sel{selection}"
    # reference_path = base_path / f"xenium_region_2-{sel}-highres_wide.png"
    matches = list((base_path / f"../../../xeniumtiles/{xenIN}/").glob(f"tile{tile}*.png"))
    # search_dir = (base_path / "../../../xeniumtiles/tiles_o20k_png_s4/").resolve()
    # print("Searching in:", search_dir)
    # print("Exists:", search_dir.exists())
    reference_path = matches[0]
    return moving_path, reference_path

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=int, default=7, help="Cell selection number.")
    parser.add_argument("--xenIN", type=str, default="", help="Xenium input folder.")
    parser.add_argument("--tile", type=str, default=7, help="Tile selection number.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "register_cells",
        help="Directory to write output figures.",
    )
    parser.add_argument("--search-dim", type=int, default=512, help="Dimension to which images are resized for search.")
    parser.add_argument("--metric", type=str, default="gpc", choices=["ncc", "ngf", "gpc"], help="Similarity metric for registration.")
    parser.add_argument("--device", type=str, default="auto", help="Device for processing (auto|cpu|cuda|mps).")
    parser.add_argument("--output-prefix", type=str, default="reg_cells", help="Prefix for output files.")
    parser.add_argument("--show", action="store_true", help="Visualize the registration results (open matplotlib).")
    args = parser.parse_args()
    return args

    
def _auto_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _str_to_translation_config(metric: str):
    if metric == "ncc":
        return NCCTranslationConfig()
    elif metric == "ngf":
        return NGFTranslationConfig()
    elif metric == "gpc":
        return GPCTranslationConfig()
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def main() -> None:
    args = _parse_args()
    print(f"_____{args}_____")
    moving_path, reference_path = _parse_image_from_selection(args.selection,args.tile,args.xenIN)
    img = ImageIO(grayscale=True).load(reference_path)
    reference = build_image_td(img)
    reference_base = reference.clone()

    img = ImageIO(grayscale=True).load(moving_path)
    moving = build_image_td(img)
    moving_base = moving.clone()

    cfg = CellSizeCfg(
        radius_hint_px=14.0,
        use_log=True,
        blob_weight=0.4,
        max_dim=1024,
        morph_radius_px=3,
        peak_rel_threshold=0.3,
    )
    moving, reference = CellUnitResize(cell_cfg=cfg)(moving, reference)
    move_scale = (moving.W / moving_base.W, moving.H / moving_base.H)
    ref_scale = (reference.W / reference_base.W, reference.H / reference_base.H)

    resize_pipe = TargetResize(target_width=args.search_dim)
    pad_pipe = CenterPad(
        scale=2,
        window_alpha=0.05,
        pad_to_pow2=False,
        outputs=["image", "box", "mask", "quad", "window"],
    )
    prepare_pipe = resize_pipe >> pad_pipe
    reference_lowres, moving_lowres = prepare_pipe(reference, moving)

    print(f"Low-resolution Reference Image Shape: {reference_lowres.image.shape}")
    print(f"Low-resolution Moving Image Shape: {moving_lowres.image.shape}")
    print(f"Moving Image Scale Factors: Width = {move_scale[0]:.4f}, Height = {move_scale[1]:.4f}")
    print(f"Reference Image Scale Factors: Width = {ref_scale[0]:.4f}, Height = {ref_scale[1]:.4f}")
    print(f"Processing on Device: {args.device}")
    

    search_params = SearchParams(
        scale_x=ScaleRange(min_scale=0.9, max_scale=1.5, step=0.01),
        scale_y=ScaleRange(min_scale=0.9, max_scale=1.5, step=0.01),
    )
    config = ExhaustiveSearchConfig(
        translation=_str_to_translation_config(args.metric),
        batch_size=4,
        progress_enabled=True,
        gradient=CDGradientConfig(
            eta=EtaConfig.from_mean(scale=0.2, norm=L2NormConfig()),
            normalize=NormalizeConfig(norm="l2", threshold=1e-3),
        ) if args.metric == "ngf" else CDGradientConfig(),
    )
    search = ExhaustiveWarpSearch(search_params, config)
   
    device = _auto_device(args.device)
    reference_lowres = reference_lowres.to(device)
    moving_lowres = moving_lowres.to(device)
    search = search.to(device)

    start_t = time.perf_counter()
    result = search(reference_lowres, moving_lowres, top_k=1, progress=True)
    end_t = time.perf_counter()
    print(f"Search took {end_t - start_t:.2f} seconds.")

    print("Estimated transformation parameters:")
    warp = result.warp
    print(f"  angle: {warp.angle.item():.2f} degrees")
    print(f"  scale_x: {warp.scale_x.item():.4f}")
    print(f"  scale_y: {warp.scale_y.item():.4f}")

    #Align full resolution images using estimated parameters
    product_pipeline = build_product_pipeline(result, move_scale=move_scale, ref_scale=ref_scale, crop_mode="union")
    product_pipeline = product_pipeline.to(device)
    registered, reference_out, original_mov = product_pipeline(
        moving_base.clone(),
        reference_base.clone(),
        moving_base.clone(),
    )

    if args.show:
        spec = OverlaySpec()

        show_comparison(
            reference_out,
            registered,
            mode="overlay",
            spec=spec,
        )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "".join([args.output_prefix,f"_sel{args.selection}"])

    out_dir = out_dir / prefix / f"metric_{args.metric}_searchdim_{args.search_dim}_{args.xenIN}_{args.tile}"

    moving_save = original_mov.clone()
    reference_save = reference_out.clone()
    registered_save = registered.clone()

    checkerboard_spec = CheckerboardSpec(
            tiles=(64, 64),
            edge_overlay=EdgeOverlaySpec(
                edge_source='mov',
                edge_threshold=0.3,
            )
        )
    overlay_spec = OverlaySpec(alpha=0.5)
    
    ProductIO(out_dir).save(
        moving=moving_save,
        reference=reference_save,
        registered=registered_save,
        search_result=result,
        overwrite=True,
        save_checkerboard=True,
        save_overlay=True,
        checkerboard_spec=checkerboard_spec,
        overlay_spec=overlay_spec,
    )
    print(f"Saved product images to: {out_dir}")


if __name__ == "__main__":
    main()
