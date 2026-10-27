"""Score held-out heart gene-expression predictions and publish slide CV folds."""
import argparse
import json
from pathlib import Path


def slide_folds(slide_ids, n_splits=10, seed=42):
    import numpy as np
    ids = np.asarray(sorted(set(slide_ids)), dtype=object)
    if len(ids) < 2 or n_splits < 2:
        raise ValueError("At least two slides and two folds are required")
    np.random.default_rng(seed).shuffle(ids)
    folds = np.array_split(ids, min(n_splits, len(ids)))
    return [{"fold": index + 1,
             "train_slides": [str(item) for item in ids if item not in set(fold)],
             "test_slides": [str(item) for item in fold]}
            for index, fold in enumerate(folds)]


def score(prediction, truth):
    import numpy as np
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 2 or not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("Prediction and truth must be matching finite [spots, genes] matrices")
    if len(truth) < 2:
        raise ValueError("At least two spots are required")
    pcc = []
    for column in range(truth.shape[1]):
        x, y = prediction[:, column], truth[:, column]
        if np.std(x) < 1e-6 or np.std(y) < 1e-6:
            pcc.append(0.0)
        else:
            pcc.append(float(np.corrcoef(x, y)[0, 1]))
    mse = np.mean((prediction - truth) ** 2, axis=0)
    return {"spots": len(truth), "genes": truth.shape[1],
            "mean_gene_pcc": float(np.mean(pcc)), "overall_mse": float(np.mean(mse)),
            "gene_pcc": pcc, "gene_mse": mse.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True, help="[spots, genes] .npy")
    parser.add_argument("--truth", type=Path, required=True, help="same ordered spots/genes .npy")
    parser.add_argument("--genes", type=Path, help="optional ordered gene names .npy")
    parser.add_argument("--slide-ids", type=Path, help="optional newline-separated slide IDs for CV manifest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    report = score(np.load(args.pred), np.load(args.truth))
    if args.genes:
        genes = [str(name) for name in np.load(args.genes, allow_pickle=True)]
        if len(genes) != report["genes"]:
            parser.error("Gene names do not match prediction columns")
        report["gene_names"] = genes
    if args.slide_ids:
        report["folds"] = slide_folds(args.slide_ids.read_text().splitlines(), seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("spots", "genes", "mean_gene_pcc", "overall_mse")}, indent=2))


if __name__ == "__main__":
    main()
