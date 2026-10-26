"""Score spatial domains with the five-seed PCA + GMM + BIC protocol.

Inputs are barcode-aligned NPZ embeddings and a headerless TSV of
``slide<TAB>barcode<TAB>label``. For one slide, ``barcode<TAB>label`` is also
accepted. Blank or ``unknown`` labels are included in clustering but excluded
from ARI, NMI, and AMI. Each slide is fitted independently.
"""
import argparse
import csv
import json
from pathlib import Path

SEEDS = (0, 1, 2, 42, 100)


def load_embeddings(path, key):
    import numpy as np

    with np.load(path) as data:
        barcodes = [str(value) for value in data["barcodes"]]
        values = np.asarray(data[key])
    if values.ndim != 2 or len(barcodes) != len(values):
        raise ValueError(f"Invalid embedding dimensions in {path}")
    if len(set(barcodes)) != len(barcodes) or not np.isfinite(values).all():
        raise ValueError(f"Duplicate barcodes or non-finite embeddings in {path}")
    return dict(zip(barcodes, values))


def load_truth(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for fields in csv.reader(stream, delimiter="\t"):
            if not fields or not any(fields):
                continue
            if len(fields) == 2:
                barcode, label = fields
                slide = "slide_1"
            elif len(fields) == 3:
                slide, barcode, label = fields
            else:
                raise ValueError("Truth TSV needs barcode/label or slide/barcode/label columns")
            rows.append((slide, barcode, label.strip()))
    if not rows or len({(slide, barcode) for slide, barcode, _ in rows}) != len(rows):
        raise ValueError("Truth TSV is empty or contains duplicate slide/barcode pairs")
    return rows


def evaluate(gene, image, truth, joint=None, seeds=SEEDS, max_components=50):
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                                 normalized_mutual_info_score)
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    by_slide = {}
    for slide, barcode, label in truth:
        if barcode in gene and barcode in image and (joint is None or barcode in joint):
            by_slide.setdefault(slide, []).append((barcode, label))
    if not by_slide:
        raise ValueError("No barcodes match both embedding files and truth TSV")

    details = []
    candidates = []
    for seed in seeds:
        for slide, rows in by_slide.items():
            fit_n = len(rows)
            if fit_n <= 10:
                raise ValueError(f"Slide {slide} needs more than 10 matching spots")
            scored = np.array([label.casefold() not in {"", "unknown"} for _, label in rows])
            if scored.sum() < 2:
                raise ValueError(f"Slide {slide} has fewer than two labeled spots")
            labels = [label for _, label in rows]
            vectors = {
                "gene": np.stack([gene[barcode] for barcode, _ in rows]),
                "image": np.stack([image[barcode] for barcode, _ in rows]),
            }
            vectors["concatenated"] = np.concatenate((vectors["gene"], vectors["image"]), axis=1)
            if joint is not None:
                vectors["joint_contextual"] = np.stack([joint[barcode] for barcode, _ in rows])
            for feature, matrix in vectors.items():
                scaled = StandardScaler().fit_transform(np.asarray(matrix, dtype=np.float64))
                components = min(max_components, fit_n - 1, scaled.shape[1])
                reduced = PCA(n_components=components, svd_solver="randomized",
                              random_state=seed).fit_transform(scaled)
                best = None
                for k in range(2, 11):
                    mixture = GaussianMixture(n_components=k, covariance_type="diag",
                                              n_init=5, max_iter=200,
                                              reg_covar=1e-6, random_state=seed).fit(reduced)
                    bic = float(mixture.bic(reduced)) if mixture.converged_ else None
                    candidates.append({"feature": feature, "seed": seed, "slide": slide,
                                       "K": k, "converged": bool(mixture.converged_), "BIC": bic})
                    if bic is not None and (best is None or (bic, k) < (best[0], best[1])):
                        best = (bic, k, mixture)
                if best is None:
                    raise RuntimeError(f"No converged GMM for {feature}, seed={seed}, slide={slide}")
                predicted = best[2].predict(reduced)
                reference = np.asarray(labels)[scored]
                assigned = predicted[scored]
                details.append({"feature": feature, "seed": seed, "slide": slide,
                                "fit_n": fit_n, "score_n": int(scored.sum()),
                                "chosen_K": best[1], "BIC": best[0],
                                "ARI": float(adjusted_rand_score(reference, assigned)),
                                "NMI": float(normalized_mutual_info_score(reference, assigned)),
                                "AMI": float(adjusted_mutual_info_score(reference, assigned))})

    per_seed = []
    summary = []
    for feature in ("gene", "image", "concatenated") + (("joint_contextual",) if joint is not None else ()):
        feature_rows = [row for row in details if row["feature"] == feature]
        for seed in seeds:
            slide_rows = [row for row in feature_rows if row["seed"] == seed]
            per_seed.append({"feature": feature, "seed": seed,
                             **{metric: float(np.mean([row[metric] for row in slide_rows]))
                                for metric in ("ARI", "NMI", "AMI")}})
        seed_rows = [row for row in per_seed if row["feature"] == feature]
        summary.append({"feature": feature, "seeds": len(seeds), "slides": len(by_slide),
                        **{f"{metric}_{suffix}": float(function([row[metric] for row in seed_rows]))
                           for metric in ("ARI", "NMI", "AMI")
                           for suffix, function in (("mean", np.mean),
                                                    ("seedSD", lambda values: np.std(values, ddof=1)
                                                     if len(values) > 1 else 0.0))}})
    return {"summary": summary, "per_seed": per_seed,
            "slide_metrics": details, "bic_candidates": candidates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-npz", type=Path, required=True)
    parser.add_argument("--image-npz", type=Path, required=True)
    parser.add_argument("--gene-key", default="gene3584")
    parser.add_argument("--image-key", default="img_embs")
    parser.add_argument("--joint-npz", type=Path)
    parser.add_argument("--joint-key", default="joint_contextual")
    parser.add_argument("--truth-tsv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(load_embeddings(args.gene_npz, args.gene_key),
                      load_embeddings(args.image_npz, args.image_key),
                      load_truth(args.truth_tsv),
                      load_embeddings(args.joint_npz, args.joint_key) if args.joint_npz else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
