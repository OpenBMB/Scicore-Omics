# Evaluation

## PatchCamelyon (PCam)

Run `eval/classify/evaluate_pcam.py` from the repository root to evaluate the
TensorFlow Datasets (`TFDS`) `patch_camelyon` **validation** split. The script
evaluates the first 32768 examples by default; use `--limit` to change the
maximum number of examples.

```bash
python eval/classify/evaluate_pcam.py \
  --tfds-dir /path/to/tfds \
  --output results/pcam_validation.json
```

Use `--model` to select a model checkpoint and `--adapter` for an optional LoRA
adapter. The output report includes the validation split description from
`eval/classify/benchmarks.py`. PCam reads its validation split directly through
TFDS; `eval/classify/prepare_manifest.py` directs PCam users to this evaluator.
