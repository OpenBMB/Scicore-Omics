import torch
torch.cuda.set_device(3)

import torch
from PIL import Image
import anndata as ad

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

# image_path = "/data2/xiaoxinyu/image.png"
image_path = "/data2/xiaoxinyu/data/PathGen/patches_test/0a6abb8a-aaf7-47a9-b48b-47c0b585a2aa_7872_20096.jpg"
gene_path = "/data2/xiaoxinyu/data/DLPFC/gene-h5ad/AAACAAGTATCTCCCA-1-1.h5ad"
image = Image.open(image_path).convert("RGB")
gene_data = ad.read_h5ad(gene_path)


model_path = "/data2/xiaoxinyu/project/model"
# model_path = "/data2/xiaoxinyu/project/finetune/output/merged_model"
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
model.eval()


msgs = [
    {
        "role": "user",
        "content": " 请描述该基因样本信息?"
    }   
]


answers = model.chat(
    image=image,
    gene_sequence=gene_data,
    msgs=msgs,
    context=None,
    processor=processor,
    tokenizer=tokenizer,
    sampling=True,
    temperature=0.7
)

print("模型回答:\n", answers)
