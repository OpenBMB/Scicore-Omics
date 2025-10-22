import os
import json
import shutil
import re

# ==== 配置部分 ====
folders = [
    "/data2/xiaoxinyu/data/book/第九版病理学（2018）_1-90.pdf-4c2e4b61-83a5-4e23-a9b9-55f746b178ac",
    "/data2/xiaoxinyu/data/book/第九版病理学（2018）_91-180.pdf-41326ad4-da5d-4924-a38e-bf29d2d89342",
    "/data2/xiaoxinyu/data/book/第九版病理学（2018）_181-270.pdf-b5e15eba-692c-4a51-a754-530090258ab2",
    "/data2/xiaoxinyu/data/book/第九版病理学（2018）_271-365.pdf-74902f3a-f315-4b69-9638-12702334fa50",
]
output_dir = "/data2/xiaoxinyu/data/book/第九版病理学（2018）_merged"
os.makedirs(output_dir, exist_ok=True)

merged_md_path = os.path.join(output_dir, "full.md")
merged_content_path = os.path.join(output_dir, "merged_content_list.json")
merged_images_dir = os.path.join(output_dir, "images")
os.makedirs(merged_images_dir, exist_ok=True)

# ==== 辅助函数 ====
def copy_images_and_fix_md(md_text, src_img_dir, prefix):
    """
    拷贝图片文件，同时在 md_text 中修复路径（仅当重名时）
    """
    used_names = set(os.listdir(merged_images_dir))
    pattern = re.compile(r'!\[.*?\]\((images/([^)]+))\)')
    
    for match in pattern.finditer(md_text):
        rel_path = match.group(1)
        img_name = match.group(2)
        src_path = os.path.join(src_img_dir, img_name)
        
        if not os.path.exists(src_path):
            continue

        dst_name = img_name
        # 若目标目录中存在重名，则加上前缀防止冲突
        if dst_name in used_names:
            name, ext = os.path.splitext(img_name)
            dst_name = f"{prefix}_{name}{ext}"
            # 替换 md 文本中的路径
            md_text = md_text.replace(f"(images/{img_name})", f"(images/{dst_name})")

        dst_path = os.path.join(merged_images_dir, dst_name)
        shutil.copy(src_path, dst_path)
        used_names.add(dst_name)

    return md_text


# ==== 合并 full.md ====
with open(merged_md_path, "w") as fout:
    for i, folder in enumerate(folders, 1):
        md_path = os.path.join(folder, "full.md")
        img_dir = os.path.join(folder, "images")
        with open(md_path, "r") as fin:
            md_text = fin.read()
        if os.path.exists(img_dir):
            md_text = copy_images_and_fix_md(md_text, img_dir, prefix=f"part{i}")
        fout.write(md_text.strip() + "\n\n")

print(f"✅ 合并 full.md 完成：{merged_md_path}")

# ==== 合并 content_list.json ====
merged_content_list = []
for folder in folders:
    content_list_path = [f for f in os.listdir(folder) if f.endswith("_content_list.json")][0]
    with open(os.path.join(folder, content_list_path), "r") as f:
        merged_content_list.extend(json.load(f))

with open(merged_content_path, "w") as f:
    json.dump(merged_content_list, f, indent=2, ensure_ascii=False)
print(f"✅ 合并 content_list.json 完成：{merged_content_path}")

# ==== 合并 layout.json（可选） ====
layout_list = []
for folder in folders:
    layout_path = os.path.join(folder, "layout.json")
    if os.path.exists(layout_path):
        with open(layout_path, "r") as f:
            layout_list.extend(json.load(f))
layout_out = os.path.join(output_dir, "merged_layout.json")
with open(layout_out, "w") as f:
    json.dump(layout_list, f, indent=2, ensure_ascii=False)
print(f"✅ 合并 layout.json 完成：{layout_out}")

print("\n🎉 所有文件已成功合并！输出文件夹：", output_dir)
