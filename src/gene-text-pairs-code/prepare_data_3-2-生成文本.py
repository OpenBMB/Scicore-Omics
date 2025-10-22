# prepare_data_generate.py
import os
import json
import re
import time
from typing import List, Dict, Tuple
from tqdm import tqdm
from openai import OpenAI

# ================= 配置 =================
RAW_DIR = "/data2/xiaoxinyu/data/STimage-1K4M/process-data/ST/desc-raw"
OUT_DIR = "/data2/xiaoxinyu/data/STimage-1K4M/process-data/ST/desc-gpt"
os.makedirs(OUT_DIR, exist_ok=True)

# 建议把 batch 调小，减少超长输出/超时概率
BATCH_SIZE = 8

# 若你一定要“每基因2–3句、每通路3–4句”，建议把 BATCH_SIZE 改成 3–5
GENE_SENTENCE_STYLE = "每个基因用1句概括"
PATHWAY_SENTENCE_STYLE = "每个通路用2句概括"

client = OpenAI(
    api_key="sk-YvgrWZx5GUDmQo4y4fE9961316D84aCaB4AcAb08A298D36f",
    base_url="https://yeysai.com/v1",
)

# ================= 工具函数 =================
DELIM_IN_START  = "<<<SAMPLE_INPUT_START file=\"{file}\">>>"
DELIM_IN_END    = "<<<SAMPLE_INPUT_END>>>"
DELIM_OUT_START = "<<<SAMPLE_OUTPUT_START file=\"{file}\">>>"
DELIM_OUT_END   = "<<<SAMPLE_OUTPUT_END>>>"

OUT_BLOCK_REGEX = re.compile(
    r"<<<SAMPLE_OUTPUT_START file=\"(?P<file>[^\"]+)\">>>\s*(?P<body>.*?)\s*<<<SAMPLE_OUTPUT_END>>>",
    re.DOTALL
)

def build_batch_prompt(batch: List[Dict]) -> str:
    # 将多个样本的输入用强定界符包住
    blocks = []
    for s in batch:
        blk = (
            DELIM_IN_START.format(file=s["file"]) + "\n" +
            f"Top10 基因: {', '.join(s['genes'])}\n" +
            f"Top3 通路: {', '.join(s['pathways']) if s['pathways'] else '无显著通路'}\n" +
            DELIM_IN_END
        )
        blocks.append(blk)
    joined = "\n\n".join(blocks)

    # 严格指令：每个样本必须以 OUTPUT_START/END 包裹；不要输出多余文本
    prompt = f"""
你是一位生物医学专家。下面将提供多个样本的输入块，每个样本的输入都由
{DELIM_IN_START.format(file='...')} 与 {DELIM_IN_END} 包围，且包含文件名。

请对每个输入样本生成一段科研风格注释，并严格按如下规则输出：
1) 对每个样本单独输出一个“输出块”，且必须使用下面的定界符包裹：
   {DELIM_OUT_START.format(file='文件名')} ... {DELIM_OUT_END}
   其中 file 的值必须与该样本输入块里的文件名完全一致。
2) 输出块内部请使用“该样本:” 开头，然后给出一段完整的中文段落，不要条目符号。
3) 内容要求（为控制长度、防止截断）：
   - {GENE_SENTENCE_STYLE} 概括 Top15 基因的主要功能/细胞定位/与样本关联（可以合并叙述，不必逐基因逐句列出）。
   - {PATHWAY_SENTENCE_STYLE} 概括 Top5 通路在该组织背景下的功能意义（若为“无显著通路”，请简单写明无显著富集）。
4) 严禁输出任何定界符以外的多余文本；不要输出示例或解释。

以下是样本输入块：
{joined}
""".strip()
    return prompt

def parse_batch_output(text: str) -> Dict[str, str]:
    """解析模型输出，返回 {file: body}"""
    results = {}
    for m in OUT_BLOCK_REGEX.finditer(text):
        fname = m.group("file")
        body  = m.group("body").strip()
        results[fname] = body
    return results

def call_gpt_with_retry(prompt: str, retries: int = 3, wait: float = 5.0):
    last_err = None
    for attempt in range(retries):
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                # 提高可返回的最大 token，避免被截断（代理是否支持取决于服务端）
                max_tokens=4096,
            )
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(wait * (1.5 ** attempt))  # 指数退避
            else:
                raise last_err

def ensure_all_written(parsed: Dict[str, str], batch: List[Dict]):
    """把解析到的文本写入文件；返回缺失文件名列表"""
    missing = []
    for s in batch:
        fname = s["file"]
        out_path = os.path.join(OUT_DIR, fname.replace(".h5ad", ".txt"))
        if fname in parsed and parsed[fname]:
            # 给正文前加一个“样本 文件名:”统一开头（若模型内部已写，会重复？不重复，因为我们要求它这么开头）
            text = parsed[fname]
            text = text.replace(fname, "").replace(fname.replace(".h5ad",""), "")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            missing.append(s)
    return missing

def single_item_prompt(item: [Dict]) -> str:
    # 为补单失败的样本构建更小的 prompt
    s = item
    inner = (
        DELIM_IN_START.format(file=s["file"]) + "\n" +
        f"组织区域: {s['region_label']}（{s['region_desc']}）\n" +
        f"Top10 基因: {', '.join(s['genes'])}\n" +
        f"Top3 通路: {', '.join(s['pathways']) if s['pathways'] else '无显著通路'}\n" +
        DELIM_IN_END
    )
    prompt = f"""
请只针对下面这个样本生成输出块，并严格用这些定界符包裹：
{DELIM_OUT_START.format(file=s["file"])} ... {DELIM_OUT_END}

内容要求与批量一致，注意控制篇幅，避免超长。

样本输入块：
{inner}
""".strip()
    return prompt

# ================= 主流程 =================
if __name__ == "__main__":
    # 读取原始 JSON
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".json")]
    samples = []
    for fname in tqdm(files, desc="加载样本信息"):
        # 若目标 txt 已存在则跳过
        out_txt = os.path.join(OUT_DIR, fname.replace(".json", ".txt"))
        if os.path.exists(out_txt):
            continue
        with open(os.path.join(RAW_DIR, fname), "r", encoding="utf-8") as f:
            samples.append(json.load(f))

    # 按批生成
    for i in tqdm(range(0, len(samples), BATCH_SIZE), desc="生成文本"):
        batch = samples[i:i + BATCH_SIZE]
        prompt = build_batch_prompt(batch)
        resp = call_gpt_with_retry(prompt)
        text = resp.choices[0].message.content

        parsed = parse_batch_output(text)
        missing = ensure_all_written(parsed, batch)

        # 对缺失的样本做单样本重试
        if missing:
            for item in missing:
                try:
                    sp = single_item_prompt(item)
                    r2 = call_gpt_with_retry(sp, retries=3, wait=6.0)
                    t2 = r2.choices[0].message.content
                    p2 = parse_batch_output(t2)
                    _ = ensure_all_written(p2, [item])
                except Exception as e:
                    # 仍失败则写一个占位提示，避免静默丢失
                    out_path = os.path.join(OUT_DIR, item["file"].replace(".h5ad", ".txt"))
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(f"【生成失败】{item['file']}：{e}\n")
