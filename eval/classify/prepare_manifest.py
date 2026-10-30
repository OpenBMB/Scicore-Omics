"""Create an auditable image/label manifest for a supported benchmark."""
import argparse
import csv
import random
from pathlib import Path

from benchmarks import BENCHMARKS

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def images(folder):
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in EXTENSIONS)


def rows_for(benchmark, root, seed=42, max_per_class=None):
    rows = []
    if benchmark == "mhist":
        with (root / "annotations.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["Partition"].strip().lower() != "test":
                    continue
                label = row["Majority Vote Label"].strip().upper()
                mapping = {"HP": 0, "SSA": 1}
                if label not in mapping:
                    raise ValueError(f"Unrecognized MHIST label: {label}")
                rows.append((root / "images" / "images" / row["Image Name"], mapping[label]))
    elif benchmark == "bach_official_400":
        candidates = [root / "Photos", root / "photos"] + list(root.rglob("Photos"))
        photos = next((folder for folder in candidates if folder.is_dir()), None)
        if photos is None:
            raise FileNotFoundError("BACH Photos directory not found")
        for folder in sorted(path for path in photos.iterdir() if path.is_dir()):
            compact = folder.name.lower().replace("_", "").replace("-", "").replace(" ", "")
            if "normal" in compact:
                label = 0
            elif "benign" in compact:
                label = 1
            elif "insitu" in compact:
                label = 2
            elif "invasive" in compact:
                label = 3
            else:
                raise ValueError(f"Unknown BACH class folder: {folder.name}")
            rows.extend((path, label) for path in images(folder))
    else:
        folders = {
            "crc_val_he_7k": ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"],
            "lc25000_lung": ["lung_image_sets/lung_aca", "lung_image_sets/lung_scc", "lung_image_sets/lung_n"],
            "lc25000_colon": ["colon_image_sets/colon_aca", "colon_image_sets/colon_n"],
        }
        if benchmark not in folders:
            raise ValueError("PCam uses its TensorFlow Datasets validation split; use evaluate_pcam.py")
        rng = random.Random(seed)
        for label, folder in enumerate(folders[benchmark]):
            class_images = images(root / folder)
            if max_per_class is not None and benchmark.startswith("lc25000"):
                class_images = sorted(rng.sample(class_images, min(max_per_class, len(class_images))))
            rows.extend((path, label) for path in class_images)
        if benchmark.startswith("lc25000"):
            rng.shuffle(rows)
    if not rows or any(not path.is_file() for path, _ in rows):
        raise ValueError("No images found or one or more manifest images are missing")
    return [{"image": str(path.relative_to(root)), "label": label} for path, label in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int)
    args = parser.parse_args()
    rows = rows_for(args.benchmark, args.data_root, args.seed, args.max_per_class)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image", "label"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} records to {args.output}")


if __name__ == "__main__":
    main()
