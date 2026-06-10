import os
import argparse
from typing import List
from time import perf_counter
import warnings

from tqdm import tqdm
import jiwer
import evaluate
import torch
import torchaudio
from transformers import pipeline
from whisper.normalizers import BasicTextNormalizer
import transformers

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

metric = evaluate.load("bleu")


def transcribe_sample_hf(path: str, greedy: bool) -> str:

    if greedy:
        # Greedy decoding
        result = pipe(path, return_timestamps=False, chunk_length_s=30,
                      generate_kwargs={"task": "transcribe", "language": "<|ar|>"})
    else:
        # Beam search decoding
        result = pipe(path, return_timestamps=False, chunk_length_s=30,
                      generate_kwargs={"task": "transcribe", "language": "<|ar|>", "num_beams": 5})

    return result['text']


def translate_sample_hf(path: str, greedy: bool) -> str:

    if greedy:
        # Greedy decoding
        result = pipe(path, return_timestamps=False, chunk_length_s=30,
                      generate_kwargs={"task": "translate", "language": "<|ar|>"})
    else:
        # Beam search decoding
        result = pipe(path, return_timestamps=False, chunk_length_s=30,
                      generate_kwargs={"task": "translate", "language": "<|ar|>", "num_beams": 5})

    return result['text']


def compute_wer(refs: List[str], preds: List[str]) -> float:
    wer = 100 * jiwer.wer(reference=refs, hypothesis=preds)
    return wer


def compute_bleu(refs: List[str], preds: List[str]) -> float:
    bleu = 100 * metric.compute(references=refs, predictions=preds)
    return bleu


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate Whisper model on ArzEn")
    
    parser.add_argument("--model-path", '-m', required=True,
                        type=str, help="Local Whisper model path or 🤗 Hub Whisper model ID")

    parser.add_argument("--mode", '-m', required=True,
                        type=str, help="Transcribe")

    parser.add_argument("--model-path", '-m', required=True,
                        type=str, help="Local Whisper model path or 🤗 Hub Whisper model ID")

    parser.add_argument("--model-path", '-m', required=True,
                        type=str, help="Local Whisper model path or 🤗 Hub Whisper model ID")

    args = parser.parse_args()

    # Setting device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Setting dtype
    torch_dtype = torch.float16 if torch.cuda.is_available(
    ) and "large" in args.model_path else torch.float32

    # Load hugging face model
    pipe = pipeline(task='automatic-speech-recognition',
                    model=args.model_path, torch_dtype=torch_dtype, device=device)

    refs = []
    preds = []
    total_inference_duration = 0.0
    total_audio_duration = 0.0
    basic_normalizer = BasicTextNormalizer(
        remove_diacritics=False, split_letters=False)

    for path, transcript in tqdm(dataset, total=len(dataset), desc=f"Transcribing {testing_dataset_name} using {os.path.basename(args.model_path)} with Hugging Face backend"):

        start = perf_counter()
        pred = transcribe_sample_hf(path=path, greedy=False)
        end = perf_counter()

        total_inference_duration += end - start
        total_audio_duration += torchaudio.info(
            path).num_frames / torchaudio.info(path).sample_rate

        refs.append(basic_normalizer(transcript))
        preds.append(basic_normalizer(pred))

    # Compute WER for normalized reference and predicted text
    wer = compute_wer(refs=refs, preds=preds)

    tqdm.write(f"Using {args.model_path}")
    tqdm.write(f"Inference time: {total_inference_duration} seconds")
    tqdm.write(f"Total audio duration: {total_audio_duration / 3600} hours")
    tqdm.write(
        f"RTF = {total_inference_duration / total_audio_duration} | { total_audio_duration / total_inference_duration}x")
    tqdm.write(f"WER = {100 * wer}%\n")
