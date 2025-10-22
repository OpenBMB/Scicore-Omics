conda activate redehist

cd /data2/xiaoxinyu/data/PathGen

# 批量下载 WSI
nohup python download_wsi.py > download_wsi.log 2>&1 &

find /data2/xiaoxinyu/data/PathGen/patches_test -type f | wc -l
find /data2/xiaoxinyu/data/PathGen/wsi_test -mindepth 1 -maxdepth 1 -type d | wc -l

# patch提取，生成jsonl格式
nohup python process_patches.py > process_patches.log 2>&1 &

# jsonl格式转换为json格式
nohup python jsonl2json.py > jsonl2json_0906.log 2>&1 &

# 拆分训练/验证集
python train_val_data.py

# 转换为英文
python translate_ch2en.py

# 微调
conda activate minicpm-v26
cd /data2/xiaoxinyu/minicpm-v-2_6/MiniCPM-V/finetune
bash finetune_ds.sh
nohup bash finetune_ds.sh > /data2/xiaoxinyu/project/output/finetune_ds.log 2>&1 &

nohup bash finetune_lora.sh > /data2/xiaoxinyu/project/output/finetune_lora.log 2>&1 &

# nvidia-smi

# 微调评估
cd /data2/xiaoxinyu/minicpm-v-2_6/MiniCPM-V/finetune/evaluate_minicpmv
### zero-shot
nohup python zero-shot.py > zero-shot.log 2>&1 &
### lmm-evaluate
CUDA_VISIBLE_DEVICES=2,3,5,6,7 nohup python evaluate_lmm.py > evaluate_lmm-lora-6000.log 2>&1 &