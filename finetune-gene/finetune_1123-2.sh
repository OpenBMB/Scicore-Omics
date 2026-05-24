

source /home/xiaoxinyu/.bashrc
conda activate /home/xiaoxinyu/miniconda3/envs/minicpm-v26

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONWARNINGS="ignore:FutureWarning" 
export CUDA_VISIBLE_DEVICES=1,2
GPUS_PER_NODE=2
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="/data2/xiaoxinyu/project/model_merged_v75" 

DATA="/data2/xiaoxinyu/project/data/DLPFC_tri_QA.json"
EVAL_DATA="/data2/xiaoxinyu/project/data/DLPFC_tri_QA.json"
FINETUNE_SCRIPT="/data2/xiaoxinyu/project/finetune-gene/finetune.py" # 使用新项目里的 finetune.py

LLM_TYPE="qwen"   
MODEL_MAX_Length=1024 # if conduct multi-images sft, please set MODEL_MAX_Length=4096

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
LOG_DIR="/data2/xiaoxinyu/project/finetune-gene/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/finetune_$(date +%Y%m%d_%H%M%S)_gene_image_text.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
torchrun $DISTRIBUTED_ARGS /data2/xiaoxinyu/project/finetune-gene/finetune.py \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path "${DATA}" \
    --eval_data_path $EVAL_DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 false \
    --bf16_full_eval false \
    --fp16 true \
    --fp16_full_eval true \
    --do_train \
    --do_eval \
    --tune_vision false \
    --tune_llm false \
    --tune_nicheformer true \
    --use_lora true \
    --lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)" \
    --max_slice_nums 9 \
    --max_steps 4000 \
    --eval_steps 1000 \
    --output_dir /data2/xiaoxinyu/project/finetune-gene/output/lora/lora-mlp-gene-1123-2 \
    --logging_dir /data2/xiaoxinyu/project/finetune-gene/output/lora/lora-mlp-gene-1123-2 \
    --logging_strategy "steps" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 5e-4 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 50 \
    --gradient_checkpointing true \
    --deepspeed /data2/xiaoxinyu/project/finetune-gene/ds_config_zero2.json \
    --report_to "tensorboard" \
 