'''
Stage2: LoRA full chain (llm+vision+resampler+gene)
'''

#!/bin/bash
set +u  
source YOUR_SOURCE_PATH  
conda activate YOUR_CONDA_ENV

# ---------------- Dist / GPU ----------------
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export NPROC_PER_NODE=6
export MASTER_PORT=$((29500 + RANDOM % 1000))

# 内存优化，防止 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# ---------------- ★ 自定义基因模块配置 ★ ----------------
# [必须修改] 请确保这个路径指向你真实的 vocab.json
export GENE_VOCAB_PATH="/model/gene_tokenizer/vocab.json"

# Nicheformer 微调开关 (true=微调部分层, false=全冻结)
export TUNE_NICHEFORMER=false

# ---------------- Paths ----------------
BASE_MODEL="/model"
DATA_DIR="YOUR_DATA_DIR"  # 替换为你的数据目录路径
OUTPUT_PATH="YOUR_OUTPUT_DIR"  # 替换为你想保存模型的目录路径
LOG_PATH="YOUR_LOG_DIR/cpt_sft_$(date +%Y%m%d_%H%M%S).log"  # 替换为你想保存日志的目录路径
# 创建日志目录
mkdir -p "$(dirname "$LOG_PATH")"

# 说明：
# - ::text 表示该 jsonl 是纯文本样本（字段一般是 "text"）
# - ::chat 表示该 jsonl 是多轮对话样本（字段一般是 "messages"）
DATASETS=(
  # "${DATA_DIR}/cpt_pmc_md_bio.jsonl"
  # "${DATA_DIR}/sft_DLPFC_STimage_tri_merged_images.jsonl" 
  "${DATA_DIR}/sft_pathology-dataset_single_merge.jsonl" 
  # "${DATA_DIR}/sft_biomedical_messages.jsonl" 
  # "${DATA_DIR}/sft_pathgen_messages.jsonl" 
  "${DATA_DIR}/sft_medical_messages.jsonl" 
  # "${DATA_DIR}/sft_medmcpa_messages.jsonl" 
  "${DATA_DIR}/sft_general_ultrachat_messages.jsonl" 
  # "${DATA_DIR}/sft_general_finevisionmax_messages_sampled.jsonl" 
  "${DATA_DIR}/sft_biopathimage.jsonl"
)
# 如果你想按比例**采样混合**（而不是用你已按比例采好的文件），可以开启下面一行并填入权重（与 DATASETS 一一对应）：
# INTERLEAVE="--interleave_prob 0.1 0.1 0.1 0.1 0.1 0.05 0.05 0.4"
INTERLEAVE=""
echo "🚀 Starting Swift SFT for MiniCPM + Gene..." | tee -a "$LOG_PATH"
echo "📂 Model: $BASE_MODEL" | tee -a "$LOG_PATH"
echo "🧬 Gene Vocab: $GENE_VOCAB_PATH" | tee -a "$LOG_PATH"


# ---------------- Run ----------------
swift sft \
  --custom_register_path "/register/my_register_qformer.py" \
  --model_type "minicpm_v2_6_gene" \
  --template "minicpm_v2_6_gene" \
  --model "$BASE_MODEL" \
  --dataset "${DATASETS[@]}" \
  $INTERLEAVE \
  --freeze_vit false \
  --freeze_aligner true \
  --dataset_shuffle true \
  --train_dataloader_shuffle true \
  --split_dataset_ratio 0.05 \
  --train_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 10 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 1 \
  --learning_rate 5e-5 \
  --lr_scheduler_type cosine \
  --report_to tensorboard \
  --gradient_accumulation_steps 16 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --max_length 2048 \
  --logging_steps 20 \
  --output_dir "$OUTPUT_PATH" \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --dataloader_num_workers 0 \
  --truncation_strategy right \
  --seed 42 \
  --deepspeed zero2 2>&1 | tee -a "$LOG_PATH"

