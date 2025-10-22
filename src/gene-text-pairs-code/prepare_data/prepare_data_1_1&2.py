# prepare_data.py
import os
import gzip
import pandas as pd
import numpy as np
import anndata as ad
import random
from tqdm import tqdm

def parse_gene_info(gene_info_path, tax_id="9606"):
    unique_pairs = set()
    with gzip.open(gene_info_path, 'rt') as f:
        for line in f:
            if line.startswith("#"): continue
            fields = line.strip().split("\t")
            if fields[0] != tax_id: continue
            symbol = fields[9]
            description = fields[5]
            if symbol != "-" and description != "-":
                unique_pairs.add((symbol, description))
    return list(unique_pairs)

def read_symbol_list(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def save_onehot_h5ad(symbol_list, target_symbol, output_path):
    expr = np.zeros((1, len(symbol_list)))
    idx = symbol_list.index(target_symbol)
    expr[0, idx] = 1
    var = pd.DataFrame(index=symbol_list)
    obs = pd.DataFrame(index=[f"sample_{target_symbol}"])
    adata = ad.AnnData(X=expr, obs=obs, var=var)
    adata.write(output_path)
    
def save_softonehot_h5ad(symbol_list, target_symbol, output_path):
    expr = np.random.normal(loc=0.0, scale=0.5, size=(1, len(symbol_list)))
    idx = symbol_list.index(target_symbol)
    expr[0, idx] = np.random.normal(loc=10.0, scale=1.0)
    var = pd.DataFrame(index=symbol_list)
    obs = pd.DataFrame(index=[f"sample_{target_symbol}"])
    adata = ad.AnnData(X=expr, obs=obs, var=var)
    adata.write(output_path)


def main():
    gene_info_path = '/data1/xiaoxinyu/project/gene-text/NCBI/gene2go/gene2go_with_symbol.gz'
    output_dir = '/data1/xiaoxinyu/project/gene_text_pairs/gene2go'
    os.makedirs(f"{output_dir}/gene-h5ad", exist_ok=True)
    # os.makedirs(f"{output_dir}/desc", exist_ok=True)

    gene_desc_pairs_all = parse_gene_info(gene_info_path)
    symbol_list_all = [g[0] for g in gene_desc_pairs_all]
    print(len(symbol_list_all)) # 363736
    gene_desc_pairs = gene_desc_pairs_all # [:10]  # 可调

    for symbol, desc in tqdm(gene_desc_pairs):
        candidates = [s for s in symbol_list_all if s != symbol]
        sampled = random.sample(candidates, 19999)
        symbol_list = [symbol] + sampled
        # save_onehot_h5ad(symbol_list, symbol, f"{output_dir}/gene-h5ad_1/{symbol}.h5ad")
        save_softonehot_h5ad(symbol_list, symbol, f"{output_dir}/gene-h5ad_2/{symbol}.h5ad")
        # with open(f"{output_dir}/desc/{symbol}.txt", 'w') as f:
        #     f.write(desc)


if __name__ == "__main__":
    main()
