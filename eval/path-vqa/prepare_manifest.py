"""Export local PathVQA parquet shards to a JSONL manifest and image directory."""
import argparse
import io
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    shards = sorted(args.parquet_dir.glob(f"{args.split}-*.parquet"))
    if not shards:
        parser.error(f"No {args.split} parquet shards found")
    from datasets import load_dataset
    from PIL import Image
    dataset = load_dataset("parquet", data_files={args.split: [str(p) for p in shards]})[args.split]
    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / f"{args.split}.jsonl"
    with manifest.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(dataset):
            value = row["image"]
            if isinstance(value, dict) and value.get("bytes"):
                image = Image.open(io.BytesIO(value["bytes"]))
            elif hasattr(value, "convert"):
                image = value
            else:
                raise ValueError(f"Unsupported image value at row {index}")
            image_name = f"{index:08d}.png"
            image.convert("RGB").save(images_dir / image_name)
            record = {"image": f"images/{image_name}", "question": row["question"],
                      "answer": row["answer"]}
            if row.get("type") is not None:
                record["type"] = row["type"]
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Exported {len(dataset)} {args.split} rows from {len(shards)} shards")


if __name__ == "__main__":
    main()
