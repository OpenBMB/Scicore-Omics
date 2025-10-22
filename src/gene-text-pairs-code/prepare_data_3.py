import anndata as ad
import pandas as pd
import gseapy as gp
from openai import OpenAI
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============ 配置 ============
input_dir = "/data2/xiaoxinyu/project/gene_text_pairs/DLPFC/gene-h5ad"
output_dir = "/data2/xiaoxinyu/project/gene_text_pairs/DLPFC/desc"
os.makedirs(output_dir, exist_ok=True)

region_map = {
    "Layer_1": "大脑皮层的分子层。",
    "Layer_2": "大脑皮层的外颗粒层。",
    "Layer_3": "大脑皮层的外锥体层。",
    "Layer_4": "大脑皮层的内颗粒层。",
    "Layer_5": "大脑皮层的内锥体层。",
    "Layer_6": "大脑皮层的多形层。",
    "WM": "大脑的白质层。"
}

client = OpenAI(
    api_key='sk-YvgrWZx5GUDmQo4y4fE9961316D84aCaB4AcAb08A298D36f',
    base_url="https://yeysai.com/v1",
)

batch_size = 64  # ✅ 批量大小


def process_file(fname):
    """单个文件处理，提取样本信息"""
    out_path = os.path.join(output_dir, fname.replace(".h5ad", ".txt"))
    if os.path.exists(out_path):
        return None  # 已经存在则跳过

    adata = ad.read_h5ad(os.path.join(input_dir, fname))
    region_label = adata.obs["region_label"].unique()[0]
    region_desc = region_map.get(region_label, "未知区域")

    # 一次性转换为 DataFrame
    df = adata.to_df()

    # Top10 基因
    gene_exp = df.mean(axis=0)
    top10_genes = gene_exp.sort_values(ascending=False).head(10).index.tolist()

    # Top3 通路
    try:
        ssgsea_res = gp.ssgsea(
            data=df.T,
            gene_sets='KEGG_2021_Human',
            sample_norm_method='rank',
            outdir=None,
            verbose=False
        )
        top3_pathways = (
            ssgsea_res.res2d.sort_values(by="ES", ascending=False)
            .head(3)["Term"].tolist()
        )
    except Exception:
        top3_pathways = []

    return {
        "file": fname,
        "region_label": region_label,
        "region_desc": region_desc,
        "genes": top10_genes,
        "pathways": top3_pathways
    }


# Step 1. 并行收集样本信息
samples = []
files = [f for f in os.listdir(input_dir) if f.endswith(".h5ad")]

with ProcessPoolExecutor() as executor:
    futures = {executor.submit(process_file, f): f for f in files}
    for future in tqdm(as_completed(futures), total=len(futures), desc="收集样本信息"):
        result = future.result()
        if result is not None:
            samples.append(result)

# Step 2. 批量送入 GPT
for i in tqdm(range(0, len(samples), batch_size), desc="生成文本"):
    batch = samples[i:i + batch_size]

    gpt_input = "\n\n".join([
        f"""样本 {j+1}:
        文件: {s['file']}
        组织区域: {s['region_label']}（{s['region_desc']}）
        Top10 基因: {', '.join(s['genes'])}
        Top3 通路: {', '.join(s['pathways']) if s['pathways'] else "无显著通路"}"""
        for j, s in enumerate(batch)
    ])

    prompt = f"""
    你是一位生物医学专家。我会给你多个样本的信息，请你逐个生成科研风格的简洁描述。

    输入信息：
    {gpt_input}

    要求：
    1. 每个样本单独输出，不要合并。
    2. 用“样本X”开头，后接描述。
    3. 每个描述遵循以下结构：
       - 首先提及组织区域。
       - 融合Top10基因功能（每个基因2–3句话）。
       - 融合Top3通路信息（每个通路3–4句话）。
       - 最终形成一段完整的科研注释。
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    batch_output = response.choices[0].message.content

    # Step 3. 按样本拆分保存
    outputs = batch_output.split("样本")
    for j, s in enumerate(batch):
        if j + 1 >= len(outputs):
            continue
        text = "样本" + outputs[j + 1].strip()
        out_path = os.path.join(output_dir, s["file"].replace(".h5ad", ".txt"))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
