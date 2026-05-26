export CUDA_VISIBLE_DEVICES=7
python train_gene_bridge_distill.py \
  --model_path MODEL_PATH \
  --data_jsonl DATA_PATH \
  --gene_vocab GENE_VOCAB \
  --out_dir OUTPUT_DIR \
  --epochs 1 \
  --batch_size 16 \
  --lr 1e-4 \
  --lambda_ce 0.2 \
  --lambda_cos 1.0 \
  --lambda_nce 1.0 \
  --temp 0.07 \
  --save_step 100