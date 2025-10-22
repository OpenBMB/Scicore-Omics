# ----------
### 数据预处理
# ----------

/data1/xiaoxinyu/project/process-data-code


# ----------
### 生成数据对
# ----------

### 全流程进行
# python -u /data1/xiaoxinyu/project/gene-text-pairs.py

### 分步进行，加快运行速度

# 1. 构造数据对
# --------------------------------------
# 1.1 伪数据
# 1.1.1 one-hot 
# 1.1.2 soft-one-hot
python -u /data1/xiaoxinyu/project/gene-text-pairs-code/prepare_data_1_1&2.py
# 1.1.3 真实表达 profile 拟合
python -u /data1/xiaoxinyu/project/gene-text-pairs-code/prepare_data_1_3.py
# --------------------------------------
# 1.2 真实数据
# 1.2.1 拆分desc/gene-h5ad + 单样本富集分析
python -u /data2/xiaoxinyu/project/gene-text-pairs-code/prepare_data_2.py
# nohup python -u /data2/xiaoxinyu/project/gene-text-pairs-code/prepare_data_2.py > /data2/xiaoxinyu/project/gene-text-pairs-code/log/0827-prepare_data_2.log 2>&1 &
# 1.2.2 细胞类型占比 + 组织区域描述 + 高表基因描述 + 代表性通路 
python -u /data2/xiaoxinyu/project/gene-text-pairs-code/prepare_data_3.py
# nohup python -u /data2/xiaoxinyu/project/gene-text-pairs-code/prepare_data_3.py > /data2/xiaoxinyu/project/gene-text-pairs-code/log/0831-prepare_data_3.log 2>&1 &
### 分步进行
python -u /data2/xiaoxinyu/project/gene-text-pairs-code/prepare_data_3-1-收集样本.py
nohup python -u /data2/xiaoxinyu/project/gene-text-pairs-code/prepare_data_3-2-生成文本.py > /data2/xiaoxinyu/project/gene-text-pairs-code/log/0831-prepare_data_3-2.log 2>&1 &


# 2. 获取基因嵌入
conda run -n nicheformer python /data2/xiaoxinyu/project/gene-text-pairs-code/batch_gene_embedding.py
# conda activate nicheformer
# nohup python -u /data2/xiaoxinyu/project/gene-text-pairs-code/batch_gene_embedding_fast.py > /data2/xiaoxinyu/project/gene-text-pairs-code/log/gene_embedding-0901.log 2>&1 &


# 3. 获取文本嵌入
conda run -n minicpm-v26 python /data2/xiaoxinyu/project/gene-text-pairs-code/batch_text_embedding.py
# conda activate minicpm-v26
# nohup python -u /data2/xiaoxinyu/project/gene-text-pairs-code/batch_text_embedding_fast.py > /data2/xiaoxinyu/project/gene-text-pairs-code/log/text_embedding-0903.log 2>&1 &


# 4. 合并结果
# 4.1 gene-text 对
python -u /data2/xiaoxinyu/project/gene-text-pairs-code/merge_results.py
# 4.2 gene-text-label 对
python -u /data2/xiaoxinyu/project/gene-text-pairs-code/merge_results_with_label.py
# nohup python -u /data1/xiaoxinyu/project/gene-text-pairs-code/merge_results.py > /data1/xiaoxinyu/project/gene-text-pairs-code/log/merge_results.log 2>&1 &


# ===============================
# 后台运行 nohup python -u prepare_data.py > prepare_data.log 2>&1 &
# 查看实时日志 tail -f prepare_data.log
# 查看进程状态  ps -p 3427086
# 中断程序 kill 182153
# ===============================


# -------------------------
### 训练任务头
# -------------------------

# conda run -n nicheformer python /data2/xiaoxinyu/nicheformer/code/emb-lp-1.py
# conda activate nicheformer
# nohup python -u /data2/xiaoxinyu/nicheformer/code/emb-lp-1.py > /data2/xiaoxinyu/nicheformer/code/log/emb-lp-1.log 2>&1 &

conda activate nicheformer
nohup python /data2/xiaoxinyu/nicheformer/code/emb-ft-1.py > /data2/xiaoxinyu/nicheformer/code/log/emb-ft-1.log 2>&1 &


# -------------------------
### 训练线性层    
# -------------------------

# 冻结模型版本
# python -u /data2/xiaoxinyu/project/train_mapping_stageB_lp.py

# 解冻模型版本
nohup python /data2/xiaoxinyu/project/train_mapping_vsepp.py > /data2/xiaoxinyu/project/logs/train_mapping_vsepp-0829.log 2>&1 &


# -------------------------
### 评估对齐效果
# --------------------------

python -u /data1/xiaoxinyu/project/eval_mapping_vsepp.py
nohup python -u /data2/xiaoxinyu/project/eval_mapping_vsepp.py > /data2/xiaoxinyu/project/logs/828-eval_mapping_vsepp.log 2>&1 &


# -------------------------
### 下游任务流程
# --------------------------

nohup python -u /data2/xiaoxinyu/project/src/multimodal_pipeline.py > /data2/xiaoxinyu/project/result/logs/0905-multimodal_pipeline.log 2>&1 &
