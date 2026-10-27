"""Benchmark prompts and label conventions transcribed from local SciCore runs."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Benchmark:
    labels: tuple[str, ...]
    prompt: str
    split: str


BENCHMARKS = {
    "mhist": Benchmark(
        ("hyperplastic polyp", "sessile serrated adenoma"),
        "You are a pathology image classification assistant.\n"
        "Classify this colorectal polyp histology image from the MHIST dataset.\n"
        "Choose exactly one label from: hyperplastic polyp, sessile serrated adenoma.\n"
        "Return only the label text, with no explanation.",
        "annotations.csv Partition=test"),
    "bach_official_400": Benchmark(
        ("normal", "benign", "in situ carcinoma", "invasive carcinoma"),
        "You are a pathology image classification assistant.\n"
        "Classify this breast histology image into exactly one label from the following list:\n"
        "normal, benign, in situ carcinoma, invasive carcinoma.\n"
        "Return only the label text, with no explanation.",
        "official 400-image folder set; no held-out split asserted"),
    "crc_val_he_7k": Benchmark(
        ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"),
        "You are a pathology AI assistant. Classify the colorectal H&E patch into one label:\n\n"
        "- ADI: Adipose tissue\n- BACK: Background/out-of-focus region\n"
        "- DEB: Tissue debris and necrosis\n- LYM: Dense lymphocyte infiltration\n"
        "- MUC: Mucin-producing glands\n- MUS: Smooth muscle fibers\n"
        "- NORM: Normal colon mucosa\n- STR: Fibrous stroma tissue\n"
        "- TUM: Tumor epithelium\n\nOnly output the class code (e.g., TUM).\n"
        "Answer format:\nTUM",
        "CRC-VAL-HE-7K class directories; all images"),
    "pcam": Benchmark(
        ("0", "1"),
        "You are a professional digital pathology AI.\n"
        "The following is a 96x96 H&E patch from lymph node (PCAM dataset).\n"
        "Classify whether metastatic tumor is present.\n\n"
        "Classes:\n0: Normal lymph node tissue, no tumor\n"
        "1: Metastatic tumor (cancer cells present)\n\nOutput only 0 or 1.",
        "TFDS patch_camelyon test; first 32768 examples in original run"),
    "lc25000_lung": Benchmark(
        ("lung adenocarcinoma", "lung squamous cell carcinoma", "benign lung tissue"),
        "<image>\nYou are a pathology image classification assistant.\n\n"
        "This is a lung histopathology image from the LC25000 dataset.\n"
        "Classify it into exactly one of the following categories:\n\n"
        "0: lung adenocarcinoma\n1: lung squamous cell carcinoma\n2: benign lung tissue\n\n"
        "Important rules:\n- Choose only one category based only on the image.\n"
        "- Do not describe the image.\n- Do not explain your reasoning.\n"
        "- Output only one digit: 0, 1, or 2.\n\nFinal answer:",
        "all images in lung_image_sets class folders; optional per-class sampling"),
    "lc25000_colon": Benchmark(
        ("colon adenocarcinoma", "benign colon tissue"),
        "<image>\nYou are a pathology image classification assistant.\n\n"
        "This is a colon histopathology image from the LC25000 dataset.\n"
        "Classify it into exactly one of the following categories:\n\n"
        "0: colon adenocarcinoma\n1: benign colon tissue\n\n"
        "Important rules:\n- Choose only one category based only on the image.\n"
        "- Do not describe the image.\n- Do not explain your reasoning.\n"
        "- Output only one digit: 0 or 1.\n\nFinal answer:",
        "all images in colon_image_sets class folders; optional per-class sampling"),
}
