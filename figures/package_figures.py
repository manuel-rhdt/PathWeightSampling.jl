#!/usr/bin/env python3
"""Package the rendered figures into one archive for download."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

FIGURES = [
    "figure_gene_expr/gene-expr-figure.png",
    "figure_gene_expr/gene-expr-figure.svg",
    "figure_gene_expr/gene-expr-figure.pdf",
    "figure_kernels/fourier_comparison.png",
    "figure_kernels/fourier_comparison.pdf",
    "figure_rates/chemotaxis.png",
    "figure_rates/chemotaxis.pdf",
    "figure_rates/chemotaxis_comparison.png",
    "figure_rates/chemotaxis_comparison.pdf",
    "figure_chemotaxis/chemotaxis_mi.png",
    "figure_chemotaxis/chemotaxis_mi.svg",
    "figure_chemotaxis/chemotaxis_mi.pdf",
]

figures_dir = Path(__file__).resolve().parent
missing = [name for name in FIGURES if not (figures_dir / name).is_file()]
if missing:
    raise SystemExit("Missing figures: " + ", ".join(missing))

archive = figures_dir.parent / "artifacts" / "pathweightsampling-figures.zip"
archive.parent.mkdir(parents=True, exist_ok=True)
with ZipFile(archive, "w", ZIP_DEFLATED) as zf:
    for name in FIGURES:
        zf.write(figures_dir / name, arcname=name)

with ZipFile(archive) as zf:
    print(f"{archive}: {len(zf.namelist())} files, {archive.stat().st_size} bytes")
    print("\n".join(zf.namelist()))
