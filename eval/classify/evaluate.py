"""Run SciCore pathology benchmark inference on a prepared image manifest."""
import argparse
import csv
import json
import re
from pathlib import Path

from benchmarks import BENCHMARKS


def parse_prediction(benchmark, output):
    text = str(output or "").strip().lower()
    labels = BENCHMARKS[benchmark].labels
    number = re.search(r"\b([0-9])\b", text)
    if number and int(number.group(1)) < len(labels):
        return int(number.group(1))
    if benchmark == "pcam":
        return 1 if "1" in text else 0
    if benchmark == "mhist":
        if any(term in text for term in ("sessile serrated", "ssa")):
            return 1
        if any(term in text for term in ("hyperplastic", "hp")):
            return 0
        return None
    if benchmark == "crc_val_he_7k":
        for index, code in enumerate(labels):
            if re.search(rf"\b{re.escape(code.lower())}\b", text):
                return index
        return None
    if benchmark.startswith("lc25000"):
        if "adenocarcinoma" in text or "aca" in text:
            return 0
        if benchmark == "lc25000_lung" and ("squamous" in text or "scc" in text):
            return 1
        if any(term in text for term in ("benign", "normal", "non-malignant")):
            return 2 if benchmark == "lc25000_lung" else 1
    for index, label in enumerate(labels):
        if label in text:
            return index
    if benchmark == "bach_official_400":
        if "in situ" in text or "in-situ" in text or "insitu" in text:
            return 2
        if "invasive" in text:
            return 3
    return None


def metrics(records, labels):
    total = len(records)
    parsed = [row for row in records if row["prediction"] is not None]
    correct = sum(row["prediction"] == row["label"] for row in records)
    confusion = [[0 for _ in labels] for _ in labels]
    for row in parsed:
        confusion[row["label"]][row["prediction"]] += 1
    per_class = {}
    for index, name in enumerate(labels):
        tp = confusion[index][index]
        fp = sum(confusion[row][index] for row in range(len(labels))) - tp
        fn = sum(confusion[index]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(confusion[index])}
    return {"total": total, "parsed": len(parsed), "correct": correct,
            "parse_rate": len(parsed) / total if total else 0.0,
            "accuracy_overall": correct / total if total else 0.0,
            "accuracy_parsed_only": correct / len(parsed) if parsed else 0.0,
            "macro_f1": sum(item["f1"] for item in per_class.values()) / len(labels),
            "per_class": per_class, "confusion_matrix_parsed_only": confusion}


def evaluate(rows, benchmark, predict):
    labels = BENCHMARKS[benchmark].labels
    records = []
    for row in rows:
        label = int(row["label"])
        if label not in range(len(labels)):
            raise ValueError(f"Label out of range: {label}")
        raw = str(predict(row["image"], BENCHMARKS[benchmark].prompt))
        records.append({"image": row["image"], "label": label, "raw_output": raw,
                        "prediction": parse_prediction(benchmark, raw)})
    return {"benchmark": benchmark, "split": BENCHMARKS[benchmark].split,
            "prompt": BENCHMARKS[benchmark].prompt,
            "metrics": metrics(records, labels), "predictions": records}


def load_model(path, adapter=None):
    import torch
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModel.from_pretrained(path, trust_remote_code=True,
                                     torch_dtype=dtype, device_map="auto")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval(), processor, tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", default="openbmb/SciCore-Omics")
    parser.add_argument("--adapter")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.benchmark == "pcam":
        parser.error("PCam uses evaluate_pcam.py")
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(not {"image", "label"} <= row.keys() for row in rows):
        parser.error("Manifest must contain image,label columns and at least one row")
    model, processor, tokenizer = load_model(args.model, args.adapter)
    import torch
    from PIL import Image

    def predict(image_name, prompt):
        path = args.data_root / image_name
        with Image.open(path) as source:
            image = source.convert("RGB")
            with torch.inference_mode():
                return model.chat(image=image, msgs=[{"role": "user", "content": prompt}],
                                  processor=processor, tokenizer=tokenizer,
                                  max_new_tokens=16, do_sample=False)

    report = evaluate(rows, args.benchmark, predict)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
