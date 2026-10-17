# Gene and pathway descriptions for spatial transcriptomics samples

These scripts document the two steps used to prepare per-sample text descriptions:

1. `collect_samples.py` reads one `.h5ad` file per sample. It selects the 15 genes with the highest mean expression and the `Term` values from the five highest-`ES` rows of GSEApy ssGSEA (`KEGG_2021_Human`, `sample_norm_method="rank"`). It writes one JSON summary per sample. The five rows are selected directly from `res2d`, without deduplication or aggregation across observations.
2. `generate_descriptions.py` groups those summaries into batches of eight and requests Chinese descriptions from an OpenAI-compatible chat API. The requested model is `gpt-4o-mini`; the endpoint can be configured, so the model actually served by a third-party endpoint cannot be determined from these scripts alone.

## Requirements

Python 3.9 or newer, `anndata`, `gseapy`, and `openai`. The first step also uses the default gene-set source resolved by GSEApy for `KEGG_2021_Human`; provide a local GMT file with `--gene-sets` if you need a fixed gene-set snapshot.

```bash
python -m pip install anndata gseapy openai
```

## Usage

```bash
python collect_samples.py \
  --input-dir /path/to/h5ad_samples \
  --output-dir /path/to/sample_summaries

export OPENAI_API_KEY="your-api-key"
python generate_descriptions.py \
  --input-dir /path/to/sample_summaries \
  --output-dir /path/to/descriptions
```

To use another OpenAI-compatible service, set `OPENAI_BASE_URL` or pass `--base-url`. Use `--model` to request another model. Both scripts skip completed outputs by default and support `--overwrite`. The collection script accepts `--workers` and `--gene-sets`; the generation script accepts `--batch-size`, `--temperature`, `--max-tokens`, and `--retries`.

## File format and prompt

Each JSON file has the same stem as its source `.h5ad` file:

```json
{
  "file": "sample.h5ad",
  "genes": ["GENE1", "GENE2"],
  "pathways": ["PATHWAY1", "PATHWAY2"]
}
```

The generation script's `build_prompt` function contains the prompt and its input/output delimiters. The prompt asks for one Chinese research-style paragraph per sample, beginning with `该样本:`, covering gene functions or cellular localization and pathway relevance. It preserves the original input labels `Top10 基因` and `Top3 通路` even though the collection step supplies up to 15 genes and five pathways, and the prompt instructions refer to Top15/Top5. Those labels are a historical inconsistency in the prompt, not different selection thresholds.

The public scripts remove machine-specific paths and embedded credentials. They validate input and model output, retry incomplete batches, and report failures with a nonzero exit status. An unsuccessful generation does not create a description file. Reproducing a particular prior run also requires its input `.h5ad` files, gene-set version, package versions, and API response; model output may vary between calls.
