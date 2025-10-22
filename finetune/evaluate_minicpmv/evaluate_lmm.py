import sys
import json
import random
import csv
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from openai import OpenAI
from peft import PeftModel

# -----------------------------
# 配置
# -----------------------------
PATHGEN_JSON = "/data2/xiaoxinyu/project/data/pathgen_train_en.json"
OUTPUT_CSV = "/data2/xiaoxinyu/project/result/1111.csv"
MODEL_DIR="/data2/xiaoxinyu/project/model"
path_to_adapter="/data2/xiaoxinyu/project/finetune/output/output__lora"

OPENAI_API_KEY = 'sk-YvgrWZx5GUDmQo4y4fE9961316D84aCaB4AcAb08A298D36f'
BASE_URL = "https://yeysai.com/v1"

sys.path.append(MODEL_DIR)

N_SAMPLES = 1

# -----------------------------
# 加载数据
# -----------------------------

with open(PATHGEN_JSON, "r") as f:
    data = json.load(f)

eval_samples = []
for item in data:
    img_path = item["image"]
    question = item["conversations"][0]["content"] 
    # question = item["conversations"][0]["content"].replace("<image>\n", "").strip()
    reference = item["conversations"][1]["content"]

    eval_samples.append({
        "id": item["id"],
        "image": img_path,
        "question": question,
        "reference": reference
    })

random.seed(42)
eval_samples = random.sample(eval_samples, min(N_SAMPLES, len(eval_samples)))


# -----------------------------
# 加载模型
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model =  AutoModel.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True
        )

model = PeftModel.from_pretrained(
    model,
    path_to_adapter,
    device_map="auto",  
    trust_remote_code=True
).eval()

processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)


# -----------------------------
# OpenAI client
# -----------------------------
client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)

def gpt4_judge(question, reference, prediction):
    prompt = f"""
你是病理学专家，请评估模型输出与参考答案的语义一致性。
问题：{question}
参考答案：{reference}
模型答案：{prediction}

请给出一个分数：
- 1 = 完全不一致
- 2 = 部分相关，但有重要错误或缺漏
- 3 = 大体一致，有小部分差异
- 4 = 高度一致，几乎相同

只输出分数，不要解释。
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",  
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        score = int(response.choices[0].message.content.strip())
    except Exception as e:
        print("GPT-4 评审失败:", e)
        score = -1
    return score


# -----------------------------
# 主循环
# -----------------------------
results = []
for sample in tqdm(eval_samples, desc="Evaluating"):
    image = Image.open(sample["image"]).convert("RGB")
    # question = sample["question"]
    question = sample["question"].replace("<image>\n", "").strip()


    # 生成答案（用 model.chat 替代 processor+generate）
    msgs = [
        {"role": "user", "content": [question]}
    ]
    
    with torch.inference_mode():
        prediction = model.chat(
            image=image,
            msgs=msgs,
            processor=processor,
            tokenizer=tokenizer,  
            max_new_tokens=1024,
        )


    # GPT-4 评审
    score = gpt4_judge(question, sample["reference"], prediction)

    results.append({
        "id": sample["id"],
        "question": question,
        "reference": sample["reference"],
        "prediction": prediction,
        "score": score
    })

# 计算分数均值
valid_scores = [r["score"] for r in results]
avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
print(f"平均分数: {avg_score:.2f} (基于 {len(valid_scores)} 个有效评分)")

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "question", "reference", "prediction", "score"])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"评估结果 {OUTPUT_CSV}")
