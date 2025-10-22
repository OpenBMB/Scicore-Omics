import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoImageProcessor,
    AutoModel,
)

# -----------------
# 配置
# -----------------
# checkpoint_dir 现在包含了模型权重和所有配置文件
# checkpoint_dir = "/data1/xiaoxinyu/huggingface/minicpm-v-2_6"
checkpoint_dir = "/data2/xiaoxinyu/minicpm-v-2_6/MiniCPM-V/finetune/output/output_minicpmv26/checkpoint-1000"
data_root = "/data2/xiaoxinyu/data/CRC-VAL-HE-7K"  # CRC-VAL-HE-7K 解压后的目录
device = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]

# class_prompts = {
#     "ADI": "A histopathology image of adipose tissue.",
#     "BACK": "A histopathology image of background region.",
#     "DEB": "A histopathology image of debris.",
#     "LYM": "A histopathology image of lymphocytes.",
#     "MUC": "A histopathology image of mucus.",
#     "MUS": "A histopathology image of smooth muscle.",
#     "NORM": "A histopathology image of normal colon mucosa.",
#     "STR": "A histopathology image of stroma tissue.",
#     "TUM": "A histopathology image of tumor epithelium."
# }

# -----------------
# Processor 加载逻辑
# -----------------
def load_processor_and_tokenizer_local():
    """
    从本地路径加载处理器和分词器。
    
    假设 checkpoint_dir 已经包含以下文件：
    - preprocessor_config.json
    - tokenizer_config.json
    - vocab.json
    - ...
    """
    print("Trying to load processor and tokenizer from local path:", checkpoint_dir)
    processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    
    print("✅ Loaded processor:", type(processor))
    print("Has image_processor:", hasattr(processor, "image_processor"))
    if hasattr(processor, "image_processor"):
        print(
            "image_feature_size:",
            getattr(processor.image_processor, "image_feature_size", None),
        )
        print(
            "patch_size:", getattr(processor.image_processor, "patch_size", None)
        )

    return processor, tokenizer


# -----------------
# 分类函数
# -----------------
def classify_image(img_path, model, processor, tokenizer):
    """用模型预测图像类别"""
    image = Image.open(img_path).convert("RGB")
    msgs = [
        {
            "role": "user",
            "content": [
                "这张组织学切片属于哪一类？可选类别是: " + ", ".join(CLASSES)
            ],
        }
    ]
    with torch.inference_mode():
        answer = model.chat(
            image=image,
            msgs=msgs,
            processor=processor,
            tokenizer=tokenizer,
            max_new_tokens=64,
        )
    return answer


def map_to_class(answer):
    """把模型输出映射到 9 个类别之一"""
    ans = answer.lower()
    for cls in CLASSES:
        if cls.lower() in ans:
            return cls
    return None


# -----------------
# 主评估逻辑
# -----------------
def evaluate(model, processor, tokenizer):
    total, correct = 0, 0
    results = []

    for cls in CLASSES:
        cls_dir = os.path.join(data_root, cls)
        if not os.path.isdir(cls_dir):
            continue

        img_files = [
            f
            for f in os.listdir(cls_dir)
            if f.endswith((".tif", ".png", ".jpg"))
        ]

        # 仅取前 200 张进行评估，以加快测试速度
        for f in tqdm(img_files[:200], desc=f"Class {cls}"):
            img_path = os.path.join(cls_dir, f)
            pred = classify_image(img_path, model, processor, tokenizer)
            pred_cls = map_to_class(pred)

            total += 1
            if pred_cls == cls:
                correct += 1
            results.append((img_path, cls, pred_cls, pred))

    acc = correct / total if total > 0 else 0
    print(f"\nZero-shot classification accuracy: {acc:.4f} ({correct}/{total})")

    with open("crc7k_zeroshot_results.txt", "w") as f:
        for r in results:
            f.write("\t".join([r[0], r[1], str(r[2]), str(r[3])]) + "\n")


# -----------------
# 主程序入口
# -----------------
if __name__ == "__main__":
    print("Loading processor/tokenizer/model...")
    # 从本地路径加载处理器和分词器
    processor, tokenizer = load_processor_and_tokenizer_local()

    # 从本地路径加载微调后的模型权重
    model = AutoModel.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    ).eval()

    evaluate(model, processor, tokenizer)