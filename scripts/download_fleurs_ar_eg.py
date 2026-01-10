"""
Download fleurs ar.eg dataset locally

Example usage:
    $ python download_fleurs_ar_eg.py
"""

import os
import math

import torch
import torchaudio
from datasets import load_dataset
from tqdm import tqdm


def build_fleurs_ar_eg(
    out_dir="../data/raw/fleurs_ar_eg",
    train_max_hours=math.inf,
    max_train_duration_sec=30.0,
):
    os.makedirs(out_dir, exist_ok=True)

    split_map = {
        "train": "train",
        "dev": "validation",
        "test": "test",
    }

    global_meta = open(os.path.join(out_dir, "metadata.tsv"), "w", encoding="utf-8")
    global_meta.write("split\tfilename\ttext\n")

    for split, hf_split in split_map.items():
        print(f"\nDownloading FLEURS ar_eg {split}…")
        ds = load_dataset("google/fleurs", "ar_eg", split=hf_split, trust_remote_code=True)

        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        total_sec = 0.0
        kept = 0

        for ex in tqdm(ds):
            audio = ex["audio"]
            text = ex["transcription"].strip()

            wav = torch.tensor(audio["array"]).unsqueeze(0)
            sr = audio["sampling_rate"]
            duration = wav.shape[1] / sr

            # Only filter TRAIN
            if split == "train":
                if duration > max_train_duration_sec:
                    continue
                if total_sec + duration > train_max_hours * 3600:
                    break

            out_name = f"{kept:06d}.wav"
            rel_path = f"{split}/{out_name}"
            out_path = os.path.join(out_dir, rel_path)

            torchaudio.save(out_path, wav, sr)
            global_meta.write(f"{split}\t{rel_path}\t{text}\n")

            total_sec += duration
            kept += 1

        print(f"{split}: {kept} files | {total_sec/3600:.2f} hours")

    global_meta.close()


build_fleurs_ar_eg()
