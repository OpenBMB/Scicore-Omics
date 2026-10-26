
"""Compute spatial domain clustering ARI from aligned image and gene embeddings.

NPZ files must contain barcodes and a 2D embedding array. Truth is TSV with
barcode and domain label columns, without a header.
"""
import argparse
import json
from pathlib import Path


def load_embeddings(path, key):
    import numpy as np
    with np.load(path) as data:
        barcodes = [str(value) for value in data["barcodes"]]
        values = np.asarray(data[key])
    if values.ndim != 2 or len(barcodes) != len(values) or len(set(barcodes)) != len(barcodes):
        raise ValueError(f"Invalid embedding data: {path}")
    return dict(zip(barcodes, values))


def evaluate(gene, image, truth, seed=42):
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import StandardScaler
    common = [(barcode, label) for barcode, label in truth if barcode in gene and barcode in image]
    labels = [label for _, label in common]
    clusters = len(set(labels))
    if clusters < 2 or len(common) < clusters:
        raise ValueError("At least two labeled domains with matching embeddings are required")
    gene_values = np.stack([gene[barcode] for barcode, _ in common])
    image_values = np.stack([image[barcode] for barcode, _ in common])
    scores = {}
    for name, values in {"gene": gene_values, "image": image_values,
                         "concatenated": np.concatenate([gene_values, image_values], axis=1)}.items():
        scaled = StandardScaler().fit_transform(values)
        predicted = KMeans(n_clusters=clusters, random_state=seed, n_init=10).fit_predict(scaled)
        scores[name] = float(adjusted_rand_score(labels, predicted))
    return {"matched_spots": len(common), "domains": clusters, "ari": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-npz", type=Path, required=True)
    parser.add_argument("--image-npz", type=Path, required=True)
    parser.add_argument("--gene-key", default="gene3584")
    parser.add_argument("--image-key", default="img_embs")
    parser.add_argument("--truth-tsv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    truth = []
    for line in args.truth_tsv.read_text().splitlines():
        if line.strip():
            barcode, label = line.split("\t", 1)
            truth.append((barcode, label))
    report = evaluate(load_embeddings(args.gene_npz, args.gene_key),
                      load_embeddings(args.image_npz, args.image_key), truth, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
