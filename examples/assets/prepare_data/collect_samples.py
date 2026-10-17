#!/usr/bin/env python3
"""Summarize genes and pathways for each spatial transcriptomics sample."""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def summarize_sample(sample_path: Path, gene_sets: str) -> dict:
    import anndata as ad
    import gseapy as gp

    expression = ad.read_h5ad(sample_path).to_df()
    if expression.empty:
        raise ValueError("expression matrix is empty")

    genes = (
        expression.mean(axis=0)
        .sort_values(ascending=False)
        .head(15)
        .index.astype(str)
        .tolist()
    )
    enrichment = gp.ssgsea(
        data=expression.T,
        gene_sets=gene_sets,
        sample_norm_method="rank",
        outdir=None,
        verbose=False,
    )
    pathways = (
        enrichment.res2d.sort_values(by="ES", ascending=False)
        .head(5)["Term"]
        .astype(str)
        .tolist()
    )
    return {"file": sample_path.name, "genes": genes, "pathways": pathways}


def save_summary(output_path: Path, summary: dict) -> None:
    temporary_path = output_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    os.replace(temporary_path, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of per-sample .h5ad files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for per-sample .json summaries")
    parser.add_argument("--gene-sets", default="KEGG_2021_Human", help="GSEApy gene-set name or GMT file")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes (default: 1)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing summaries")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    if not args.input_dir.is_dir():
        print(f"Input directory does not exist: {args.input_dir}", file=sys.stderr)
        return 2

    paths = sorted(args.input_dir.glob("*.h5ad"))
    if not paths:
        print(f"No .h5ad files found in {args.input_dir}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        path for path in paths
        if args.overwrite or not (args.output_dir / f"{path.stem}.json").exists()
    ]
    failed = 0
    completed = 0

    if args.workers == 1:
        for path in pending:
            try:
                summary = summarize_sample(path, args.gene_sets)
                save_summary(args.output_dir / f"{path.stem}.json", summary)
                completed += 1
            except Exception as exc:
                failed += 1
                print(f"Failed {path.name}: {exc}", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(summarize_sample, path, args.gene_sets): path
                for path in pending
            }
            for future in as_completed(futures):
                path = futures[future]
                try:
                    summary = future.result()
                    save_summary(args.output_dir / f"{path.stem}.json", summary)
                    completed += 1
                except Exception as exc:
                    failed += 1
                    print(f"Failed {path.name}: {exc}", file=sys.stderr)

    print(f"Written: {completed}; skipped: {len(paths) - len(pending)}; failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
