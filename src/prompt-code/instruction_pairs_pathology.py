import os
import re
import json
from tqdm import tqdm
from openai import OpenAI
import time

# ============ 配置部分 ============

# client = OpenAI(
#     api_key="sk-8c939933ad18451f91a04ea559641697",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )

client = OpenAI(
    api_key="sk-5a687022325b4129a6e9892c63209a1d",
    base_url="https://api.deepseek.com",
)


# 教材路径（MinerU 处理结果）
root_dir = "/data2/xiaoxinyu/data/book/textbook_of_pathology.pdf-2cbcd0e0-47f5-486c-93b7-5418e9055648"
input_md = os.path.join(root_dir, "full.md")
image_dir = os.path.join(root_dir, "images")

# 输出路径
output_path = "/data2/xiaoxinyu/project/data/textbook_of_pathology/instruction_pairs_pathology.jsonl"

# 每篇文本生成的问题数量
num_questions = 4


# ============ Prompt 模板 ============

QUESTION_PROMPT = """
你是一名资深的病理学与组织学研究者，正在为一个多模态大模型构建生物医学指令微调数据。

请根据以下教材或论文内容，生成多样化的高质量问题。

要求：
1. 问题必须基于原文内容，不可编造；
2. 问题类型需多样：形态学理解、疾病机制、图像判读、诊断推理、比较分析、临床应用；
3. 确保问题逻辑严谨，能引导模型输出完整的医学解释；
4. 输出为 JSON 数组，每个元素形如：[{{"question": "..."}}]
5. 每次生成 {num_questions} 个问题。

文本：
\"\"\"{text}\"\"\"
"""


IMAGE_QA_PROMPT = """
你是一名病理学与组织学专家。请基于以下图像相关内容（包含图像上下文和图注）生成高质量问题。

要求：
1. 问题需围绕图像或图注展开，例如结构、形态、功能或临床意义；
2. 保证问题逻辑清晰，符合病理学知识；
3. 输出为 JSON 数组：[{{"question": "..."}}, ...]
4. 每次生成 {num_questions} 个问题。

上下文：
{text}
图注：
{caption}
"""


ANSWER_PROMPT = """
你是一名病理学助理，请基于以下文本内容，为问题提供逻辑清晰、简洁准确的回答。
问题：
"{question}"
文本：
\"\"\"{text}\"\"\"
请输出纯文本回答。
"""


# ============ 工具函数 ============
def split_markdown_with_sections(md_text):
    """按章节分块，同时检测图片与上下文"""
    sections, current_section, buffer = [], "引言", []
    for line in md_text.splitlines():
        if re.match(r"^#{1,3}\s", line):
            if buffer:
                sections.append({"section": current_section, "text": "\n".join(buffer)})
                buffer = []
            current_section = line.strip("# ").strip()
        else:
            buffer.append(line)
    if buffer:
        sections.append({"section": current_section, "text": "\n".join(buffer)})
    return sections


def extract_images_from_text(text):
    """从 markdown 块中抽取图像路径及其上下文、图注"""
    pattern = re.compile(r'!\[\]\((.*?)\)')
    matches = pattern.findall(text)
    results = []
    for m in matches:
        img_path = os.path.join(image_dir, os.path.basename(m))
        # 找出图像的上下文与可能的caption
        lines = text.splitlines()
        idx = next((i for i, l in enumerate(lines) if m in l), None)
        caption = ""
        context = ""
        if idx is not None:
            caption = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
            context = "\n".join(lines[max(0, idx - 3):idx]).strip()
        results.append({"image": img_path, "caption": caption, "context": context})
    return results


def safe_api_call(func, *args, max_retries=3, **kwargs):
    """简单的重试逻辑"""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"⚠️ API 调用失败({attempt+1}/{max_retries})：{e}")
            time.sleep(2)
    return None


def generate_questions(prompt):
    response = safe_api_call(
        client.chat.completions.create,
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )
    if not response:
        return []
    raw_output = response.choices[0].message.content.strip()
    raw_output = re.sub(r"^```(json)?", "", raw_output, flags=re.MULTILINE)
    raw_output = re.sub(r"```$", "", raw_output, flags=re.MULTILINE)
    raw_output = raw_output.strip()
    try:
        questions = json.loads(raw_output)
    except Exception:
        questions = [{"question": q.strip()} for q in re.split(r"[\n；;]", raw_output) if q.strip()]
    return questions


def generate_answer(question, text):
    prompt = ANSWER_PROMPT.format(question=question, text=text)
    response = safe_api_call(
        client.chat.completions.create,
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    if not response:
        return ""
    return response.choices[0].message.content.strip()


# ============ 主流程 ============
def main():
    print(f"📘 正在读取教材文件：{input_md}")
    text = open(input_md, "r", encoding="utf-8").read()
    sections = split_markdown_with_sections(text)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fout = open(output_path, "w", encoding="utf-8")
    id_counter = 1820

    print(f"🧩 检测到 {len(sections)} 个章节块。")
    
    # start_idx, end_idx = 120, 130
    # sections = sections[start_idx:end_idx]
    # print(f"🔍 仅测试第 {start_idx}-{end_idx} 个章节块，共 {len(sections)} 段")

    start_idx = 290
    sections = sections[start_idx:]

    for sec in tqdm(sections, desc="Processing sections"):
        section_name = sec["section"]
        chunk_text = sec["text"]

        # 1️⃣ 生成纯文本问题
        q_prompt = QUESTION_PROMPT.format(text=chunk_text, num_questions=num_questions)
        q_list = generate_questions(q_prompt)

        for q in q_list:
            a = generate_answer(q["question"], chunk_text)
            record = {
                "id": f"{id_counter:06d}",
                # "section": section_name,
                "conversations": [
                    {"role": "user", "content": q["question"]},
                    {"role": "assistant", "content": a},
                ],
                # "task": "pathology_text_instruction",
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            id_counter += 1

        # 2️⃣ 生成图像相关问题
        images = extract_images_from_text(chunk_text)
        for img in images:
            img_prompt = IMAGE_QA_PROMPT.format(
                text=img["context"], caption=img["caption"], num_questions=num_questions
            )
            q_list = generate_questions(img_prompt)
            for q in q_list:
                a = generate_answer(q["question"], img["context"] + "\n" + img["caption"])
                record = {
                    "id": f"{id_counter:06d}",
                    # "section": section_name,
                    "conversations": [
                        {"role": "user", "content": f"<image>\n{q['question']}"},
                        {"role": "assistant", "content": a},
                    ],
                    "image": img["image"],
                    # "caption": img["caption"],
                    # "task": "pathology_image_instruction",
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                id_counter += 1

    fout.close()
    print(f"\n✅ 已生成 {id_counter-1} 条指令样本，保存到：{output_path}")


if __name__ == "__main__":
    main()