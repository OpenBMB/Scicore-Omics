"""Evaluate the first N examples of the TFDS patch_camelyon test split."""
import argparse
import json
from pathlib import Path

from benchmarks import BENCHMARKS
from evaluate import load_model, parse_prediction, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfds-dir", type=Path, required=True)
    parser.add_argument("--model", default="openbmb/SciCore-Omics")
    parser.add_argument("--adapter")
    parser.add_argument("--limit", type=int, default=32768)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import tensorflow_datasets as tfds
    import torch
    from PIL import Image
    model, processor, tokenizer = load_model(args.model, args.adapter)
    dataset = tfds.load("patch_camelyon", data_dir=str(args.tfds_dir),
                        split="test", as_supervised=True)
    records = []
    for index, (image_tensor, label_tensor) in enumerate(dataset.take(args.limit)):
        image = Image.fromarray(image_tensor.numpy()).convert("RGB")
        with torch.inference_mode():
            raw = model.chat(image=image,
                             msgs=[{"role": "user", "content": BENCHMARKS["pcam"].prompt}],
                             processor=processor, tokenizer=tokenizer,
                             max_new_tokens=2, do_sample=False)
        records.append({"index": index, "label": int(label_tensor.numpy()),
                        "raw_output": str(raw), "prediction": parse_prediction("pcam", raw)})
    report = {"benchmark": "pcam", "split": BENCHMARKS["pcam"].split,
              "prompt": BENCHMARKS["pcam"].prompt,
              "metrics": metrics(records, BENCHMARKS["pcam"].labels),
              "predictions": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
