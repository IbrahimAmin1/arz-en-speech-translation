"""
Download MGB-3 dataset locally

Example usage:
    $ python build_mgb3.py
"""

import os

import torch
import torchaudio
from datasets import load_dataset
from tqdm import tqdm


def build_mgb3_ar(
    out_dir="mgb3_ar",
):
    os.makedirs(out_dir, exist_ok=True)

    split_map = {
        "train": "train",
        "dev": "validation",
        "test": "test",
    }

    meta = open(os.path.join(out_dir, "metadata.tsv"), "w", encoding="utf-8")
    meta.write("split\tfilename\ttext\n")

    for split, hf_split in split_map.items():
        print(f"\nDownloading MGB-3 Arabic {split}…")
        ds = load_dataset("MohamedRashad/MGB-3-Arabic", split=hf_split)

        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        kept = 0
        total_sec = 0.0

        for ex in tqdm(ds):
            audio = ex["audio"]
            text = ex["text"].strip()

            wav = torch.tensor(audio["array"]).unsqueeze(0)
            sr = audio["sampling_rate"]
            duration = wav.shape[1] / sr

            fname = f"{kept:06d}.wav"
            rel_path = f"{split}/{fname}"
            out_path = os.path.join(out_dir, rel_path)

            torchaudio.save(out_path, wav, sr)
            meta.write(f"{split}\t{rel_path}\t{text}\n")

            total_sec += duration
            kept += 1

        print(f"{split}: {kept} files | {total_sec/3600:.2f} hrs")

    meta.close()

build_mgb3_ar()