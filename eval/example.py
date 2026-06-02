import argparse
import torch
import anndata as ad
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Run SciCore-Omics inference.")
    parser.add_argument("--model_path", type=str, default="openbmb/SciCore-Omics")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--gene_path", type=str, default=None)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Please describe the biological state of this sample."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.image_path is None and args.gene_path is None:
        raise ValueError("Please provide at least one of --image_path or --gene_path.")

    image = None
    gene_data = None

    if args.image_path is not None:
        image = Image.open(args.image_path).convert("RGB")

    if args.gene_path is not None:
        gene_data = ad.read_h5ad(args.gene_path)

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    msgs = [
        {
            "role": "user",
            "content": args.prompt
        }
    ]

    with torch.no_grad():
        response = model.chat(
            image=image,
            gene_sequence=gene_data,
            msgs=msgs,
            context=None,
            processor=processor,
            tokenizer=tokenizer,
            sampling=True,
            temperature=args.temperature,
        )

    print(response)


if __name__ == "__main__":
    main()
