"""PathVQA automatic metrics and optional local-image inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import string

SYNONYMS = {"yeah": "yes", "y": "yes", "yea": "yes", "nope": "no", "n": "no",
            "haemorrhage": "hemorrhage", "bleeding": "hemorrhage",
            "neutrophils": "neutrophil", "two": "2", "three": "3"}


def normalize(value):
    text = str(value or "").lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return SYNONYMS.get(text, text)


def question_type(row):
    value = str(row.get("type", "")).lower()
    if value in {"closed", "closed-ended", "yesno", "yes/no"}:
        return "closed"
    if value in {"open", "open-ended", "open_set", "open-set"}:
        return "open"
    if normalize(row["answer"]) in {"yes", "no"}:
        return "closed"
    return "closed" if re.match(r"^(is|are|do|does|did|was|were|can|could|should|has|have|had)\b",
                                row["question"].strip().lower()) else "open"


def score(row, prediction):
    reference = normalize(row["answer"])
    candidate = normalize(prediction)
    kind = question_type(row)
    if kind == "closed":
        if reference in {"yes", "no"}:
            candidate = next((word for word in candidate.split() if word in {"yes", "no"}), candidate)
        return kind, float(candidate == reference)
    reference_tokens = Counter(reference.split())
    predicted_tokens = Counter(candidate.split())
    overlap = sum(min(count, predicted_tokens[token]) for token, count in reference_tokens.items())
    total = sum(reference_tokens.values())
    return kind, overlap / total if total else 0.0


def evaluate(rows, predict):
    records = []
    class_counts = {}
    for row in rows:
        prediction = str(predict(row))
        kind, value = score(row, prediction)
        records.append({**row, "prediction": prediction, "type": kind, "score": value})
        reference, candidate = normalize(row["answer"]), normalize(prediction)
        class_counts.setdefault(reference, {"tp": 0, "actual": 0, "pred": 0})["actual"] += 1
        class_counts.setdefault(candidate, {"tp": 0, "actual": 0, "pred": 0})["pred"] += 1
        if reference == candidate:
            class_counts[reference]["tp"] += 1
    def average(values):
        return 100 * sum(values) / len(values) if values else None
    closed = [row["score"] for row in records if row["type"] == "closed"]
    opened = [row["score"] for row in records if row["type"] == "open"]
    weighted = 0.0
    for counts in class_counts.values():
        tp, actual, predicted = counts["tp"], counts["actual"], counts["pred"]
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        weighted += f1 * actual
    summary = {"num_total": len(records), "num_closed": len(closed), "num_open": len(opened),
               "closed_accuracy_pct": average(closed), "open_recall_pct": average(opened),
               "overall_mixed_score_pct": average([row["score"] for row in records]),
               "weighted_f1_exact_pct": 100 * weighted / len(records) if records else None}
    return {"summary": summary, "predictions": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="JSONL rows with image, question, answer, optional type")
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--model", default="openbmb/SciCore-Omics")
    parser.add_argument("--adapter", help="Optional PathVQA LoRA checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows or any(not {"image", "question", "answer"} <= row.keys() for row in rows):
        parser.error("Each JSONL row needs image, question, and answer")
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                                                 torch_dtype=dtype, device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def predict(row):
        instruction = ("Answer 'yes' or 'no' only." if question_type(row) == "closed"
                       else "Answer with a single word or short phrase. Do not add explanations or punctuation.")
        with Image.open(args.image_root / row["image"]) as source:
            image = source.convert("RGB")
            with torch.inference_mode():
                return model.chat(image=image,
                                  msgs=[{"role": "user", "content": f"{row['question']}\n{instruction}"}],
                                  context=None, processor=processor, tokenizer=tokenizer,
                                  sampling=False, temperature=0.0, max_new_tokens=32)
    report = evaluate(rows, predict)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
