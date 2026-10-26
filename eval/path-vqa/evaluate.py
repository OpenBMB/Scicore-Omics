"""Evaluate PathVQA questions with SciCore-Omics.

Input JSONL records: {"image": "relative/path.png", "question": "...", "answer": "..."}.
Closed questions use normalized yes/no accuracy; open questions use token recall.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import string


def normalize(value):
    text = str(value).lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def score(prediction, answer):
    reference = normalize(answer)
    candidate = normalize(prediction)
    if reference in {"yes", "no"}:
        words = candidate.split()
        selected = next((word for word in words if word in {"yes", "no"}), None)
        return "closed_accuracy", float(selected == reference)
    reference_tokens = Counter(reference.split())
    predicted_tokens = Counter(candidate.split())
    total = sum(reference_tokens.values())
    overlap = sum(min(count, predicted_tokens[token]) for token, count in reference_tokens.items())
    return "open_recall", overlap / total if total else 0.0


def evaluate(rows, predict):
    results = []
    for row in rows:
        prediction = str(predict(row["image"], row["question"]))
        metric, value = score(prediction, row["answer"])
        results.append({**row, "prediction": prediction, "metric": metric, "score": value})
    summary = {}
    for metric in ("closed_accuracy", "open_recall"):
        values = [row["score"] for row in results if row["metric"] == metric]
        summary[metric] = {"count": len(values), "mean": sum(values) / len(values) if values else None}
    return {"summary": summary, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--model", default="openbmb/SciCore-Omics")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows or any(not {"image", "question", "answer"} <= row.keys() for row in rows):
        parser.error("Input must contain image, question, and answer in each JSONL record")

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                      torch_dtype=dtype, device_map="auto").eval()

    def predict(image_name, question):
        path = Path(image_name)
        if not path.is_absolute():
            path = args.image_root / path
        instruction = "Answer yes or no only." if normalize(question).startswith(("is ", "are ", "does ", "do ")) else "Answer with a short phrase."
        with Image.open(path) as source:
            image = source.convert("RGB")
            with torch.inference_mode():
                return model.chat(image=image,
                                  msgs=[{"role": "user", "content": f"{question}\n{instruction}"}],
                                  context=None, processor=processor, tokenizer=tokenizer,
                                  sampling=False, max_new_tokens=32)

    report = evaluate(rows, predict)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

