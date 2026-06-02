import torch
from PIL import Image
import anndata as ad
import scipy.sparse as sp
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer, AutoProcessor

image_path = "/your/path/to/image/example.png"
gene_path = "/your/path/to/image/example.h5ad"
image = Image.open(image_path).convert("RGB")
gene_data = ad.read_h5ad(gene_path)

model_path = "/your/path/to/model"
# path_to_adapter=" "
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
# model = PeftModel.from_pretrained(model,path_to_adapter,device_map="auto", trust_remote_code=True)
model.eval()


msgs = [
    {
        "role": "user",
        "content": "请描述样本信息"
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

