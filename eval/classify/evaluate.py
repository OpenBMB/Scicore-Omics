
"""Zero-shot image classification with SciCore-Omics.

Input CSV columns: image, label. Labels must match one of --classes.
Paths in the image column are relative to --image-root unless absolute.
"""
import argparse
import csv
import json
import re
from pathlib import Path


def parse_prediction(response, classes):
    text = str(response).strip().casefold()
    text = re.sub(r"^(answer|label)\s*[:：]\s*", "", text).strip(" .,:;\n")
    matches = [label for label in classes if text == label.casefold()]
    return matches[0] if len(matches) == 1 else None


def evaluate(rows, predict, classes):
    results = []
    for row in rows:
        predicted = parse_prediction(predict(row["image"]), classes)
        results.append({"image": row["image"], "label": row["label"],
                        "prediction": predicted, "correct": predicted == row["label"]})
    correct = sum(item["correct"] for item in results)
    return {"total": len(results), "correct": correct,
            "accuracy": correct / len(results) if results else 0.0,
            "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--model", default="openbmb/SciCore-Omics")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.classes)) != len(args.classes):
        parser.error("--classes must be unique")
    with args.csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any("image" not in row or "label" not in row for row in rows):
        parser.error("CSV must contain image and label columns and at least one row")
    if any(row["label"] not in args.classes for row in rows):
        parser.error("CSV contains a label absent from --classes")

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                      torch_dtype=dtype, device_map="auto").eval()
    prompt = ("Classify this pathology image into exactly one of: "
              + ", ".join(args.classes) + ". Reply with the label only.")

    def predict(image_name):
        path = Path(image_name)
        if not path.is_absolute():
            path = args.image_root / path
        with Image.open(path) as source:
            image = source.convert("RGB")
            with torch.inference_mode():
                return model.chat(image=image, msgs=[{"role": "user", "content": prompt}],
                                  context=None, processor=processor, tokenizer=tokenizer,
                                  sampling=False, max_new_tokens=16)

    report = evaluate(rows, predict, args.classes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Accuracy: {report['correct']}/{report['total']} = {report['accuracy']:.4f}")


if __name__ == "__main__":
    main()
