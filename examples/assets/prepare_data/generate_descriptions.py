#!/usr/bin/env python3
"""Generate per-sample Chinese descriptions from gene and pathway summaries."""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


INPUT_START = '<<<SAMPLE_INPUT_START file="{file}">>>'
INPUT_END = "<<<SAMPLE_INPUT_END>>>"
OUTPUT_START = '<<<SAMPLE_OUTPUT_START file="{file}">>>'
OUTPUT_END = "<<<SAMPLE_OUTPUT_END>>>"
OUTPUT_PATTERN = re.compile(
    r'<<<SAMPLE_OUTPUT_START file="(?P<file>[^"]+)">>>\s*'
    r'(?P<body>.*?)\s*<<<SAMPLE_OUTPUT_END>>>',
    re.DOTALL,
)


def load_sample(json_path: Path) -> dict:
    with json_path.open(encoding="utf-8") as handle:
        sample = json.load(handle)
    if not isinstance(sample, dict):
        raise ValueError(f"{json_path.name}: expected a JSON object")

    filename = sample.get("file")
    if (
        not isinstance(filename, str)
        or not filename.endswith(".h5ad")
        or "/" in filename
        or "\\" in filename
        or Path(filename).stem != json_path.stem
    ):
        raise ValueError(f"{json_path.name}: file must match the JSON filename and contain no directory")
    if not isinstance(sample.get("genes"), list) or not all(
        isinstance(gene, str) for gene in sample["genes"]
    ):
        raise ValueError(f"{json_path.name}: genes must be a list of strings")
    if not isinstance(sample.get("pathways"), list) or not all(
        isinstance(pathway, str) for pathway in sample["pathways"]
    ):
        raise ValueError(f"{json_path.name}: pathways must be a list of strings")
    return sample


def build_prompt(samples: list[dict]) -> str:
    blocks = []
    for sample in samples:
        blocks.append(
            INPUT_START.format(file=sample["file"])
            + "\n"
            + f"Top10 基因: {', '.join(sample['genes'])}\n"
            + f"Top3 通路: {', '.join(sample['pathways']) if sample['pathways'] else '无显著通路'}\n"
            + INPUT_END
        )
    joined = "\n\n".join(blocks)
    return f"""
你是一位生物医学专家。下面将提供多个样本的输入块，每个样本的输入都由
{INPUT_START.format(file='...')} 与 {INPUT_END} 包围，且包含文件名。

请对每个输入样本生成一段科研风格注释，并严格按如下规则输出：
1) 对每个样本单独输出一个“输出块”，且必须使用下面的定界符包裹：
   {OUTPUT_START.format(file='文件名')} ... {OUTPUT_END}
   其中 file 的值必须与该样本输入块里的文件名完全一致。
2) 输出块内部请使用“该样本:” 开头，然后给出一段完整的中文段落，不要条目符号。
3) 内容要求（为控制长度、防止截断）：
   - 每个基因用1句概括 概括 Top15 基因的主要功能/细胞定位/与样本关联（可以合并叙述，不必逐基因逐句列出）。
   - 每个通路用2句概括 概括 Top5 通路在该组织背景下的功能意义（若为“无显著通路”，请简单写明无显著富集）。
4) 严禁输出任何定界符以外的多余文本；不要输出示例或解释。

以下是样本输入块：
{joined}
""".strip()


def parse_output(text: str) -> dict[str, str]:
    return {
        match.group("file"): match.group("body").strip()
        for match in OUTPUT_PATTERN.finditer(text)
    }


def valid_description(text: str) -> bool:
    return text.startswith(("该样本:", "该样本：")) and "【生成失败】" not in text


def save_description(output_dir: Path, filename: str, text: str) -> None:
    output_path = output_dir / f"{Path(filename).stem}.txt"
    temporary_path = output_path.with_suffix(".txt.tmp")
    temporary_path.write_text(text, encoding="utf-8")
    os.replace(temporary_path, output_path)


def generate_batch(client, samples: list[dict], args: argparse.Namespace) -> list[dict]:
    pending = samples
    for attempt in range(args.retries):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": build_prompt(pending)}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            parsed = parse_output(response.choices[0].message.content or "")
            remaining = []
            for sample in pending:
                description = parsed.get(sample["file"], "")
                if valid_description(description):
                    save_description(args.output_dir, sample["file"], description)
                else:
                    remaining.append(sample)
            pending = remaining
        except Exception as exc:
            print(f"Request failed (attempt {attempt + 1}/{args.retries}): {exc}", file=sys.stderr)
        if not pending:
            break
        if attempt + 1 < args.retries:
            time.sleep(5.0 * (1.5 ** attempt))
    return pending


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of per-sample .json summaries")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated .txt descriptions")
    parser.add_argument("--model", default="gpt-4o-mini", help="Requested chat model name")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"), help="Optional OpenAI-compatible API URL")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing valid descriptions")
    args = parser.parse_args()
    if args.batch_size < 1 or args.retries < 1 or args.max_tokens < 1:
        parser.error("--batch-size, --retries and --max-tokens must be positive")
    return args


def main() -> int:
    args = parse_args()
    if not args.input_dir.is_dir():
        print(f"Input directory does not exist: {args.input_dir}", file=sys.stderr)
        return 2

    json_paths = sorted(args.input_dir.glob("*.json"))
    if not json_paths:
        print(f"No .json files found in {args.input_dir}", file=sys.stderr)
        return 2

    samples = []
    skipped = 0
    for json_path in json_paths:
        try:
            sample = load_sample(json_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"Invalid input {json_path.name}: {exc}", file=sys.stderr)
            return 2
        output_path = args.output_dir / f"{json_path.stem}.txt"
        if not args.overwrite and output_path.exists() and valid_description(
            output_path.read_text(encoding="utf-8").strip()
        ):
            skipped += 1
            continue
        samples.append(sample)

    if not samples:
        print(f"Written: 0; skipped: {skipped}; failed: 0")
        return 0

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY must be set to generate descriptions", file=sys.stderr)
        return 2
    try:
        from openai import OpenAI
    except ImportError as exc:
        print(f"The openai package is required: {exc}", file=sys.stderr)
        return 2

    client_options = {"api_key": api_key}
    if args.base_url:
        client_options["base_url"] = args.base_url
    client = OpenAI(**client_options)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for start in range(0, len(samples), args.batch_size):
        failed.extend(generate_batch(client, samples[start:start + args.batch_size], args))
    for sample in failed:
        print(f"No valid description for {sample['file']}", file=sys.stderr)
    print(f"Written: {len(samples) - len(failed)}; skipped: {skipped}; failed: {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
