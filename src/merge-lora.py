from transformers import AutoModel
from peft import PeftModel

MODEL_DIR = "/data2/xiaoxinyu/project/model"
path_to_adapter = "/data2/xiaoxinyu/project/finetune/output/output__lora"

# 加载基础模型
model = AutoModel.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True
)

# 加载LoRA适配器
model = PeftModel.from_pretrained(
    model,
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True
)

# 合并LoRA参数到base模型，并卸载adapter
model = model.merge_and_unload()

# 现在 model 已经是合并后的完整模型，可以直接保存
save_path = "/data2/xiaoxinyu/merged_model"
model.save_pretrained(save_path)
print(f"完整权重已保存到 {save_path}")
