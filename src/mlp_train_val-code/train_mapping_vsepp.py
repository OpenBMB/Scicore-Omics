### 🔧 文件 1：train_mapping_vsepp.py

import torch
from torch.utils.data import DataLoader
import swanlab
from model_utils import GeneTextDataset, MappingNet, train

if __name__ == "__main__":
    torch.manual_seed(42)
    swanlab.init(project="gene-text-alignment", name="train_mapping_B2")

    data_path = "/data2/xiaoxinyu/project/data/gene_text_pairs/DLPFC/gene-with-complex-text/gene_text_pairs_ft20_gpt_all.pt"
    dataset = GeneTextDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = MappingNet().cuda()
    train(model, dataloader, epochs=100, lr=1e-3, loss_type="vse") # vse / nce / mse /cosine

    torch.save(model.state_dict(), "/data2/xiaoxinyu/project/model-vsepp/mapping_model_stageB-test-DLPFC_ft20_gpt_vse_all.pt")
    print("✅ 模型训练完成并保存！")
