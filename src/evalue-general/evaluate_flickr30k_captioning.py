# ======================================================
# evaluate_flickr30k_captioning.py
# 任务：用 MiniCPM-V 生成 Flickr30k 图片描述并计算 BLEU
# ======================================================

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from flickr30k_dataset import Flickr30kDataset
from transformers import AutoModel
from PIL import Image
import json
import sacrebleu

# ========== 配置 ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "/data2/xiaoxinyu/project/model"   # 替换本地模型路径
image_dir = "/data2/xiaoxinyu/data/flickr30k_images/flickr30k_images/flickr30k_images"
jsonl_path = "/data2/xiaoxinyu/data/flickr30k_images/ln_flickr30k_val_captions_multi.jsonl"

# ========== 1. 加载模型 ==========
print("🚀 加载模型中...")
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
print("✅ 模型加载完成！")

# ========== 2. 加载数据 ==========
dataset = Flickr30kDataset(image_dir=image_dir, jsonl_path=jsonl_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
print(f"✅ 数据集加载完成，共 {len(dataset)} 张图片")

# ========== 3. 生成描述 ==========
preds = {}
for batch in tqdm(dataloader, desc="🔹生成图片描述中"):
    image = batch["image"][0].to(device)
    image_id = batch["image_id"][0]

    try:
        # ✅ MiniCPM-V 正确调用方式
        caption = model.chat(image, "Describe this image in detail.")
        preds[image_id] = caption.strip()
    except Exception as e:
        print(f"[警告] {image_id} 生成失败: {e}")

# 保存结果
with open("flickr30k_preds.json", "w", encoding="utf-8") as f:
    json.dump(preds, f, ensure_ascii=False, indent=2)

print(f"✅ 已保存 {len(preds)} 条预测到 flickr30k_preds.json")

# ========== 4. 计算 BLEU ==========
if len(preds) == 0:
    print("❌ 未生成任何预测结果，请检查模型.chat是否返回字符串。")
    exit()

refs = {}
with open(jsonl_path, "r") as f:
    for line in f:
        obj = json.loads(line)
        refs.setdefault(obj["image_id"], []).append(obj["caption"].strip())

img_ids = [iid for iid in preds.keys() if iid in refs]
hyps = [preds[iid] for iid in img_ids]
refs_list = [refs[iid] for iid in img_ids]

bleu = sacrebleu.corpus_bleu(hyps, list(zip(*refs_list)))
print("\n🎯 评估结果：")
print(f"BLEU score: {bleu.score:.2f}")
print(f"评估样本数: {len(hyps)}")
print("输出结果文件: flickr30k_preds.json")
