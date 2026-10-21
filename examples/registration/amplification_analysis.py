"""Panel-wide transcription changes with EGFR amplification, and ecDNA candidates.

Reads the outputs of ``fish_to_xenium_cells.py`` (``cells.csv``, ``spots.csv``)
plus the Xenium output folder, and writes into the same folder:

Genes
    ``genes_vs_amplification.csv``: for every panel gene, how its expression
    changes with the EGFR / CEP7 FISH ratio, over all cells and within Xenium
    cell clusters. The within-cluster number separates "follows EGFR
    amplification" from "tumour cells differ from normal cells".

ecDNA candidates
    ``ecdna_cells.csv``: per-cell amplification class from the FISH spots:
    not amplified, chromosome 7 gain, clustered amplification (few compact
    EGFR foci, HSR-like) or ecDNA-like (many dispersed EGFR foci, or EGFR foci
    outside the nucleus). These are heuristic candidates at the FISH image
    resolution, not a diagnosis.

``amplification_report.html`` / ``.pdf`` show both.

Run:
    uv run --no-sync python examples/registration/amplification_analysis.py \
        --cells outputs/.../refine/cells/cells.csv \
        --xenium-dir /path/to/xenium_output
"""

from __future__ import annotations

import argparse
import base64
import io
import textwrap
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import mannwhitneyu, rankdata

INK = "#1c2328"
RED = "#c0392b"
CLASS_ORDER = ["not amplified", "chr7 gain", "clustered (HSR-like)", "ecDNA-like"]
CLASS_COLORS = {"not amplified": "#b8c0c6", "chr7 gain": "#5d8aa8", "clustered (HSR-like)": "#e0a33a", "ecDNA-like": RED}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", type=Path, required=True, help="cells.csv from fish_to_xenium_cells.py (spots.csv must sit next to it)")
    p.add_argument("--xenium-dir", type=Path, required=True)
    p.add_argument("--clustering", default="gene_expression_graphclust", help="Folder under analysis/clustering/ to stratify by.")
    p.add_argument("--min-cluster-cells", type=int, default=300, help="Clusters with fewer usable cells are left out of the within-cluster score.")
    p.add_argument("--top", type=int, default=15, help="Genes shown per direction in the figures.")
    p.add_argument("--amp-ratio", type=float, default=2.0, help="EGFR / CEP7 spot ratio counted as amplified.")
    p.add_argument("--amp-min-egfr", type=int, default=3, help="... and at least this many EGFR spots.")
    p.add_argument("--amp-min-egfr-no-cep7", type=int, default=4, help="EGFR spots counted as amplified when the nucleus has no CEP7 spot.")
    p.add_argument("--chr7-gain-cep7", type=int, default=3, help="Non-amplified nuclei with at least this many CEP7 spots are chr7 gain.")
    p.add_argument("--ecdna-min-egfr", type=int, default=4, help="Amplified nuclei need at least this many EGFR spots to be ecDNA-like ...")
    p.add_argument("--ecdna-dispersion", type=float, default=0.5,
                   help="... and spots spread at least this far (RMS distance from their mean / nucleus radius), "
                        "or at least one EGFR spot outside the nucleus. Uniform spread over a disc gives about 0.71.")
    p.add_argument("--output-dir", type=Path, default=None, help="Default: folder of cells.csv")
    return p.parse_args()


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def usable_cells(path: Path) -> pd.DataFrame:
    cells = pd.read_csv(path)
    keep = (cells["nucleus_area_fish_px"] > 0) & cells["in_fish"] & ~cells["saturated"]
    if "debris" in cells:
        keep &= ~cells["debris"]
    return cells[keep & np.isfinite(cells["egfr_cep7_nuc_ratio"])].reset_index(drop=True)


def expression(h5_path: Path, cell_ids: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Counts per 1,000 gene transcripts, log1p, for the given cells (rows) and panel genes (columns)."""
    with h5py.File(h5_path, "r") as h:
        m = h["matrix"]
        names = [n.decode() for n in m["features/name"][:]]
        kinds = [t.decode() for t in m["features/feature_type"][:]]
        barcodes = pd.Index([b.decode() for b in m["barcodes"][:]])
        indptr, indices, data = m["indptr"][:], m["indices"][:], m["data"][:]
    genes = [i for i, k in enumerate(kinds) if k == "Gene Expression"]
    cols = barcodes.get_indexer(cell_ids)
    if (cols < 0).any():
        raise ValueError("Some cells are missing from cell_feature_matrix.h5.")
    dense = np.zeros((len(cols), len(names)), dtype=np.float32)
    for row, c in enumerate(cols):
        sl = slice(indptr[c], indptr[c + 1])
        dense[row, indices[sl]] = data[sl]
    dense = dense[:, genes]
    total = dense.sum(axis=1, keepdims=True)
    return np.log1p(1000.0 * dense / np.maximum(total, 1)), [names[i] for i in genes]


def clusters(xenium_dir: Path, name: str, cell_ids: pd.Series) -> pd.Series:
    path = xenium_dir / "analysis" / "clustering" / name / "clusters.csv"
    table = pd.read_csv(path).set_index("Barcode")["Cluster"]
    return table.reindex(cell_ids).fillna(-1).astype(int).reset_index(drop=True)


# --------------------------------------------------------------------------
# Genes vs amplification
# --------------------------------------------------------------------------


def _rank_corr(x_rank: np.ndarray, y_rank: np.ndarray) -> np.ndarray:
    """Spearman correlation of one ranked vector against every column of a ranked matrix."""
    xc = x_rank - x_rank.mean()
    yc = y_rank - y_rank.mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (xc @ yc) / (np.sqrt((xc**2).sum()) * np.sqrt((yc**2).sum(axis=0)))


def gene_table(expr: np.ndarray, genes: list[str], amp: np.ndarray, cluster: pd.Series, min_cells: int) -> tuple[pd.DataFrame, list]:
    ranks = rankdata(expr, axis=0).astype(np.float32)
    rho_all = _rank_corr(rankdata(amp).astype(np.float32), ranks)

    num = np.zeros(len(genes))
    den = 0
    used = []
    for name, idx in cluster.groupby(cluster).groups.items():
        idx = np.asarray(idx)
        if len(idx) < min_cells:
            continue
        r = _rank_corr(rankdata(amp[idx]).astype(np.float32), rankdata(expr[idx], axis=0).astype(np.float32))
        num += len(idx) * np.nan_to_num(r)
        den += len(idx)
        used.append((name, len(idx)))
    rho_within = num / max(den, 1)

    lo, hi = np.quantile(amp, [0.1, 0.9])
    low, high = expr[amp <= lo], expr[amp >= hi]
    raw_low, raw_high = np.expm1(low).mean(axis=0), np.expm1(high).mean(axis=0)
    log2fc = np.log2((raw_high + 0.1) / (raw_low + 0.1))
    p = mannwhitneyu(high, low, axis=0).pvalue
    order = np.argsort(p)
    q = np.empty_like(p)
    q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    table = pd.DataFrame({
        "gene": genes,
        "rho_all_cells": rho_all,
        "rho_within_clusters": rho_within,
        "per1000_bottom10pct": raw_low,
        "per1000_top10pct": raw_high,
        "log2fc_top_vs_bottom10pct": log2fc,
        "mannwhitney_p": p,
        "bh_q": np.minimum(q, 1.0),
    }).sort_values("rho_within_clusters", ascending=False)
    return table, used


def genes_figure(table: pd.DataFrame, top: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.scatter(table["rho_all_cells"], table["rho_within_clusters"], s=12, color="#9aa3a9", linewidths=0)
    pick = pd.concat([table.head(top), table.tail(top)])
    ax.scatter(pick["rho_all_cells"], pick["rho_within_clusters"], s=16, color=INK)
    for _, r in pick.iterrows():
        ax.annotate(r.gene, (r.rho_all_cells, r.rho_within_clusters), fontsize=7, xytext=(3, 2), textcoords="offset points")
    if "EGFR" in set(table.gene):
        e = table[table.gene == "EGFR"].iloc[0]
        ax.scatter([e.rho_all_cells], [e.rho_within_clusters], s=60, color=RED, zorder=3)
    lim = float(np.nanmax(np.abs(table[["rho_all_cells", "rho_within_clusters"]].to_numpy()))) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color="#c9ced2", lw=1, ls=":")
    ax.axhline(0, color="#c9ced2", lw=1)
    ax.axvline(0, color="#c9ced2", lw=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Spearman rho with EGFR/CEP7 FISH ratio, all cells")
    ax.set_ylabel("Spearman rho within Xenium cell clusters (weighted mean)")
    ax.set_title("Every panel gene (EGFR in red). Genes far from the dotted line\n"
                 "mostly reflect cell-type differences, not amplification.")
    return fig


def gene_heatmap(expr: np.ndarray, genes: list[str], amp: np.ndarray, table: pd.DataFrame, top: int) -> plt.Figure:
    bins = pd.qcut(amp, 10, labels=False)
    chosen = list(table.head(top).gene) + list(table.tail(top).gene[::-1])
    idx = [genes.index(g) for g in chosen]
    means = np.vstack([expr[bins == b][:, idx].mean(axis=0) for b in range(10)]).T
    z = (means - means.mean(axis=1, keepdims=True)) / np.maximum(means.std(axis=1, keepdims=True), 1e-6)
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(chosen) + 1.8), constrained_layout=True)
    im = ax.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_yticks(range(len(chosen)), chosen, fontsize=8)
    ax.set_xticks(range(10), [f"{10 * b}-{10 * b + 10}%" for b in range(10)], rotation=45, ha="right", fontsize=8)
    ax.axhline(top - 0.5, color=INK, lw=1)
    ax.set_xlabel("cells grouped by EGFR/CEP7 FISH ratio (lowest to highest 10%)")
    ax.set_title(f"Top {top} genes rising and falling with amplification (within-cluster rank)")
    fig.colorbar(im, ax=ax, label="mean expression, z-score across groups", shrink=0.6)
    return fig


def cluster_figure(amp: np.ndarray, cluster: pd.Series) -> plt.Figure:
    bins = pd.qcut(amp, 10, labels=False)
    share = pd.crosstab(bins, cluster, normalize="index")
    keep = share.max(axis=0).sort_values(ascending=False).index[:12]
    share = share[keep].assign(other=1 - share[keep].sum(axis=1))
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    bottom = np.zeros(len(share))
    colors = plt.get_cmap("tab20").colors
    for i, col in enumerate(share.columns):
        ax.bar(share.index, share[col], bottom=bottom, color="#d5d9dc" if col == "other" else colors[i % 20], label=str(col))
        bottom += share[col].to_numpy()
    ax.set_xticks(range(10), [f"{10 * b}-{10 * b + 10}%" for b in range(10)], rotation=45, ha="right")
    ax.set_xlabel("cells grouped by EGFR/CEP7 FISH ratio (lowest to highest 10%)")
    ax.set_ylabel("share of cells")
    ax.set_title("Xenium cluster make-up of each amplification group")
    ax.legend(title="cluster", fontsize=7, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    return fig


# --------------------------------------------------------------------------
# ecDNA candidates
# --------------------------------------------------------------------------


def dispersion(spots: pd.DataFrame, cells: pd.DataFrame) -> pd.Series:
    """RMS distance of a nucleus's EGFR spots from their mean, divided by the nucleus radius."""
    s = spots[(spots.channel == "egfr") & (spots.location == "nucleus") & spots.cell_id.isin(cells.cell_id)]
    g = s.groupby("cell_id")
    cx = g.fish_x.transform("mean")
    cy = g.fish_y.transform("mean")
    rms = np.sqrt(((s.fish_x - cx) ** 2 + (s.fish_y - cy) ** 2).groupby(s.cell_id).mean())
    radius = np.sqrt(cells.set_index("cell_id")["nucleus_area_fish_px"] / np.pi)
    out = (rms / radius.reindex(rms.index)).where(g.size() >= 3)
    return out.reindex(cells.cell_id).reset_index(drop=True)


def classify(cells: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    e, c = cells["egfr_spots_nuc"], cells["cep7_spots_nuc"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = e / c.where(c > 0)
    amplified = ((c > 0) & (ratio >= args.amp_ratio) & (e >= args.amp_min_egfr)) | ((c == 0) & (e >= args.amp_min_egfr_no_cep7))
    ecdna = amplified & (e >= args.ecdna_min_egfr) & (
        (cells["egfr_dispersion"] >= args.ecdna_dispersion) | (cells["egfr_spots_extranuc"] >= 1)
    )
    label = np.where(ecdna, "ecDNA-like", np.where(amplified, "clustered (HSR-like)",
                     np.where(c >= args.chr7_gain_cep7, "chr7 gain", "not amplified")))
    return pd.Series(pd.Categorical(label, categories=CLASS_ORDER))


def ecdna_figure(cells: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    counts = cells["amp_class"].value_counts().reindex(CLASS_ORDER)
    ax = axes[0]
    ax.bar(range(4), counts, color=[CLASS_COLORS[k] for k in CLASS_ORDER])
    for i, v in enumerate(counts):
        ax.text(i, v, f"{v:,}\n({v / counts.sum():.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(4), CLASS_ORDER)
    ax.set_ylabel("cells")
    ax.set_title("FISH amplification class per cell")
    ax.set_ylim(0, counts.max() * 1.18)

    ax = axes[1]
    g = cells.groupby("amp_class", observed=False)["EGFR_transcripts"]
    mean, sem = g.mean().reindex(CLASS_ORDER), g.sem().reindex(CLASS_ORDER)
    ax.bar(range(4), mean, yerr=1.96 * sem, capsize=4, color=[CLASS_COLORS[k] for k in CLASS_ORDER])
    ax.set_xticks(range(4), CLASS_ORDER)
    ax.set_ylabel("mean EGFR transcripts per cell (Xenium)")
    ax.set_title("EGFR transcription by FISH class (bars: 95% CI)")

    ax = axes[2]
    for k in CLASS_ORDER:
        sub = cells[cells.amp_class == k]
        ax.scatter(sub.fish_x, sub.fish_y, s=0.6 if k == "not amplified" else 2.5, color=CLASS_COLORS[k], label=k, linewidths=0)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("FISH x (px)")
    ax.set_ylabel("FISH y (px)")
    ax.set_title("Where each class sits in the FISH image")
    ax.legend(markerscale=5, frameon=False, fontsize=8, loc="lower left")
    return fig


# --------------------------------------------------------------------------


def _uri(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def write_report(out_dir: Path, sections: list[tuple[str, list[str], list[tuple[str, plt.Figure]], pd.DataFrame | None]]) -> None:
    parts = []
    for title, notes, figs, table in sections:
        parts.append(f"<h2>{title}</h2><ul>{''.join(f'<li>{n}</li>' for n in notes)}</ul>")
        for caption, fig in figs:
            parts.append(f"<h3>{caption}</h3><img src='{_uri(fig)}' alt='{caption}'>")
        if table is not None:
            parts.append(f"<div class='scroll'>{table.to_html(index=False, float_format=lambda v: f'{v:.3g}', border=0)}</div>")
    (out_dir / "amplification_report.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>EGFR amplification analysis</title>
<style>
 body {{ font: 15px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #1c2328; background: #f6f7f8; margin: 0; }}
 main {{ max-width: 1150px; margin: 0 auto; padding: 24px 16px 48px; }}
 h1 {{ font-size: 24px; }} h2 {{ font-size: 19px; margin-top: 36px; }} h3 {{ font-size: 15px; }}
 table {{ border-collapse: collapse; background: #fff; font-size: 13px; }}
 th, td {{ border: 1px solid #dde1e4; padding: 4px 8px; text-align: right; }} th {{ background: #eef1f3; }}
 td:first-child, th:first-child {{ text-align: left; }}
 img {{ max-width: 100%; height: auto; background: #fff; border: 1px solid #dde1e4; }}
 .scroll {{ overflow-x: auto; }}
</style></head><body><main><h1>EGFR amplification: transcription changes and ecDNA candidates</h1>
{''.join(parts)}</main></body></html>""", encoding="utf-8")
    with PdfPages(out_dir / "amplification_report.pdf") as pdf:
        for title, notes, figs, table in sections:
            page = plt.figure(figsize=(11, 8.5))
            text = [title, ""] + [textwrap.fill(f"- {n}", 140, subsequent_indent="  ") for n in notes]
            if table is not None:
                text += ["", table.to_string(index=False, float_format=lambda v: f"{v:.3g}")]
            page.text(0.04, 0.96, "\n".join(text), va="top", family="monospace", fontsize=7)
            pdf.savefig(page)
            plt.close(page)
            for _, fig in figs:
                pdf.savefig(fig)


def main() -> None:
    args = _parse_args()
    out_dir = args.output_dir or args.cells.parent
    cells = usable_cells(args.cells)
    spots = pd.read_csv(args.cells.parent / "spots.csv", dtype={"cell_id": str})
    amp = cells["egfr_cep7_nuc_ratio"].to_numpy()
    print(f"usable cells: {len(cells):,}")

    # Genes
    expr, genes = expression(args.xenium_dir / "cell_feature_matrix.h5", cells["cell_id"])
    cluster = clusters(args.xenium_dir, args.clustering, cells["cell_id"])
    table, used = gene_table(expr, genes, amp, cluster.fillna(-1), args.min_cluster_cells)
    table.to_csv(out_dir / "genes_vs_amplification.csv", index=False, float_format="%.5g")
    show = pd.concat([table.head(args.top), table.tail(args.top)])[
        ["gene", "rho_within_clusters", "rho_all_cells", "log2fc_top_vs_bottom10pct", "bh_q"]]
    print(f"genes: {len(genes)}; clusters used for the within-cluster score: {len(used)}")
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # ecDNA candidates
    cells["egfr_dispersion"] = dispersion(spots, cells)
    cells["amp_class"] = classify(cells, args)
    cells["cluster"] = cluster
    cols = ["cell_id", "x_centroid", "y_centroid", "fish_x", "fish_y", "cluster", "egfr_spots_nuc", "cep7_spots_nuc",
            "egfr_spots_extranuc", "egfr_dispersion", "egfr_cep7_nuc_ratio", "EGFR_transcripts", "amp_class"]
    cells[cols].to_csv(out_dir / "ecdna_cells.csv", index=False, float_format="%.4f")
    summary = cells.groupby("amp_class", observed=False).agg(
        cells=("cell_id", "size"), mean_EGFR_transcripts=("EGFR_transcripts", "mean"),
        median_EGFR_spots=("egfr_spots_nuc", "median"), median_CEP7_spots=("cep7_spots_nuc", "median"),
        cells_with_extranuclear_EGFR=("egfr_spots_extranuc", lambda s: int((s > 0).sum())),
    ).reset_index()
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # How well do the ecDNA criteria separate cells at this resolution?
    many = cells[cells["egfr_spots_nuc"] >= args.ecdna_min_egfr]["egfr_dispersion"].dropna()
    passes = float((many >= args.ecdna_dispersion).mean()) if len(many) else float("nan")
    extra = cells.groupby("amp_class", observed=False)["egfr_spots_extranuc"].apply(lambda v: (v > 0).mean()).reindex(CLASS_ORDER)
    ecd = cells[cells.amp_class == "ecDNA-like"]
    by_disp = int(((ecd.egfr_dispersion >= args.ecdna_dispersion) & (ecd.egfr_spots_extranuc == 0)).sum())
    by_extra = int(((ecd.egfr_dispersion.fillna(0) < args.ecdna_dispersion) & (ecd.egfr_spots_extranuc > 0)).sum())
    diagnostics = [
        f"Dispersion: {passes:.0%} of nuclei with at least {args.ecdna_min_egfr} EGFR spots pass the {args.ecdna_dispersion:g} cut "
        f"(median {many.median():.2f}). If this is close to 100%, dispersion is not separating ecDNA from HSR at this resolution; "
        "the split is then mostly driven by the spot count.",
        "Cells with at least one extranuclear EGFR spot: " + ", ".join(f"{k} {v:.0%}" for k, v in extra.items())
        + ". The rate in non-amplified cells is the background from segmentation and focus mismatch.",
        f"ecDNA-like cells called by dispersion alone: {by_disp:,}; by extranuclear spots alone: {by_extra:,}; "
        f"by both: {len(ecd) - by_disp - by_extra:,}.",
    ]
    for d in diagnostics:
        print(d)

    gene_notes = [
        f"{len(cells):,} usable cells, {len(genes)} panel genes. Expression is transcripts per 1,000 of the cell's gene transcripts (log1p).",
        "Amplification is the EGFR/CEP7 nuclear FISH intensity ratio (continuous, the measure that tracks EGFR transcripts best).",
        f"'All cells' correlation mixes cell types. 'Within clusters' averages the correlation inside each Xenium "
        f"{args.clustering} cluster with at least {args.min_cluster_cells} usable cells ({len(used)} clusters), "
        "so it asks whether a gene changes with amplification among similar cells.",
        "log2 fold change and the Mann-Whitney test compare the top and bottom 10% of cells by FISH ratio; q is Benjamini-Hochberg.",
        "Chromosome positions are not in the Xenium panel file, so genes co-amplified with EGFR on 7p11 are not marked automatically.",
    ]
    ecdna_notes = diagnostics + [
        f"Amplified: EGFR/CEP7 spot ratio of at least {args.amp_ratio:g} with at least {args.amp_min_egfr} EGFR spots, "
        f"or at least {args.amp_min_egfr_no_cep7} EGFR spots when the nucleus has no CEP7 spot.",
        f"ecDNA-like: amplified, at least {args.ecdna_min_egfr} EGFR spots, and either spread across the nucleus "
        f"(dispersion at least {args.ecdna_dispersion:g}; a uniform spread over the nucleus gives about 0.71) "
        "or at least one EGFR spot inside the cell but outside the nucleus.",
        "Clustered (HSR-like): amplified but with fewer or compact EGFR foci.",
        f"chr7 gain: not amplified, with at least {args.chr7_gain_cep7} CEP7 spots.",
        "These classes come from spot counts on a 0.49 um/pixel image, where adjacent EGFR copies merge into one spot. "
        "Treat them as candidates to inspect, not as calls; every threshold is a command-line option.",
    ]
    sections = [
        ("Transcription changes with EGFR amplification", gene_notes, [
            ("Correlation with amplification, all cells vs within clusters", genes_figure(table, args.top)),
            ("Expression across amplification groups", gene_heatmap(expr, genes, amp, table, args.top)),
            ("Cell-type make-up of the amplification groups", cluster_figure(amp, cluster.fillna(-1))),
        ], show),
        ("ecDNA candidates", ecdna_notes, [("Amplification classes", ecdna_figure(cells))], summary),
    ]
    write_report(out_dir, sections)
    print(f"outputs in {out_dir}: genes_vs_amplification.csv, ecdna_cells.csv, amplification_report.html, .pdf")


if __name__ == "__main__":
    main()
