# eval_mapping_vsepp.py

import torch
from torch.utils.data import DataLoader
import swanlab
from model_utils import GeneTextDataset, MappingNet, evaluate

if __name__ == "__main__":
    torch.manual_seed(42)
    swanlab.init(project="gene-text-alignment", name="eval_mapping_vsepp")

    data_path = "/data2/xiaoxinyu/project/gene_text_pairs/DLPFC/gene-with-complex-text/gene_text_pairs_ft20_gpt.pt"
    dataset = GeneTextDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    model = MappingNet().cuda()
    model.load_state_dict(torch.load("/data2/xiaoxinyu/project/model-vsepp/mapping_model_stageB-test-DLPFC_ft20_gpt_vse.pt"))
    
    # evaluate(model, dataloader)
    evaluate(model, dataloader, method="umap", save_path="/data2/xiaoxinyu/project/logs/tsne_alignment-test-DLPFC_ft20_gpt_vse.png")

    print("✅ 模型评估完成！")