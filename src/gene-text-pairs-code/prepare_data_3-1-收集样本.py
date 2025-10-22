# prepare_data_collect.py
import anndata as ad
import pandas as pd
import gseapy as gp
import os
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============ 配置 ============
input_dir = "/data2/xiaoxinyu/data/STimage-1K4M/process-data/ST/gene_h5ad"
output_dir = "/data2/xiaoxinyu/data/STimage-1K4M/process-data/ST/desc-raw"
# input_dir = "/data2/xiaoxinyu/project/gene_text_pairs/DLPFC/gene-h5ad"
# output_dir = "/data2/xiaoxinyu/project/gene_text_pairs/DLPFC/desc-raw"
os.makedirs(output_dir, exist_ok=True)

def process_file(fname):
    out_path = os.path.join(output_dir, fname.replace(".h5ad", ".json"))
    if os.path.exists(out_path):
        return None

    adata = ad.read_h5ad(os.path.join(input_dir, fname))

    df = adata.to_df()

    # Top10 基因
    gene_exp = df.mean(axis=0)
    top10_genes = gene_exp.sort_values(ascending=False).head(15).index.tolist()

    # Top3 通路
    try:
        ssgsea_res = gp.ssgsea(
            data=df.T,
            gene_sets="KEGG_2021_Human",
            sample_norm_method="rank",
            outdir=None,
            verbose=False
        )
        top3_pathways = (
            ssgsea_res.res2d.sort_values(by="ES", ascending=False)
            .head(5)["Term"].tolist()
        )
    except Exception:
        top3_pathways = []

    sample = {
        "file": fname,
        "genes": top10_genes,
        "pathways": top3_pathways
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)

    return sample


if __name__ == "__main__":
    files = [f for f in os.listdir(input_dir) if f.endswith(".h5ad")]

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, f): f for f in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="收集样本信息"):
            _ = future.result()
