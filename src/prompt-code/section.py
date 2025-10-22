import os
import re

# ====== 配置部分 ======
root_dir = "/data2/xiaoxinyu/data/book/textbook_of_pathology.pdf-2cbcd0e0-47f5-486c-93b7-5418e9055648"
input_md = os.path.join(root_dir, "full.md")
target_image = "images/8b302c5d5104e307dcb795cc2d0ebe71e9de6f00502864b243727765a0263b4f.jpg"  # 目标图片路径（可相对路径）

# ====== 读取文件 ======
with open(input_md, "r", encoding="utf-8") as f:
    md_text = f.read()

# ====== 分块函数 ======
def split_markdown_with_sections(md_text):
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

sections = split_markdown_with_sections(md_text)

# ====== 搜索图片所在章节 ======
found = False
for idx, sec in enumerate(sections):
    if target_image in sec["text"]:
        print(f"✅ 找到图片所在章节：")
        print(f"章节序号：{idx}")
        print(f"章节名称：{sec['section']}")
        print(f"前后上下文预览：\n{sec['text'][:500]}...\n")
        found = True
        break

if not found:
    print("❌ 未在任何章节中找到该图片路径，请检查图片名是否正确或是否有相对路径差异。")
