import errno
import os
from typing import List, Tuple
import math

import numpy as np
import pandas as pd
import torchaudio
from datasets import load_dataset


# Fine-tuning datasets
def load_fleurs_ar_eg(split: str, threshold: float = math.inf) -> List[Tuple[str, str, float]]:
    fleurs = load_dataset("google/fleurs", "ar_eg",
                          split="validation" if split == "dev" else split, trust_remote_code=True).to_pandas()
    df = fleurs.loc[:, ["path", "transcription"]]

    paths = df["path"]
    paths = [os.path.join(os.path.dirname(path), split, os.path.basename(
        path)) for path in paths]  # Fix some path error in fleurs dataset
    transcriptions = df["transcription"]

    durations = [torchaudio.info(
        path).num_frames / torchaudio.info(path).sample_rate for path in paths]

    # Whisper's Feature Extractor will truncate samples more than 30 seconds during data loading which may lead to subpar results
    # Since there are only a few samples more than 30 seconds in the "train" and "validation" splits, I decided to just remove them
    filtered_data = [(path, transcription, duration) for path, transcription, duration in zip(
        paths, transcriptions, durations) if duration < 30.0]

    # Calculate total duration and select samples until it reaches threshold
    total_duration = 0
    filtered_data_with_threshold = []

    for data in filtered_data:
        if total_duration + data[2] > threshold * 60 * 60:
            break
        filtered_data_with_threshold.append(data)
        total_duration += data[2]

    print(
        f"Total duration of Fleurs {split} split of {len(filtered_data_with_threshold)} files: {total_duration / 60 / 60} hrs")

    return filtered_data_with_threshold


def load_custom_yt_dataset(path: str) -> List[Tuple[str, str, float]]:

    if not os.path.exists(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    df = pd.read_csv(path, sep='\t')
    df.replace('', np.nan, inplace=True)
    df.dropna(inplace=True)

    # Filter the DataFrame for the specified conditions
    filtered_df = df[
        (df['spoken_language'] == 'ar: Arabic') &
        (df['written_language'] == 'arb_Arab') &
        (df['spoken_dialect'] == 'EGY') &
        (df['written_dialect'] == 'EGY') &
        (df['duration'] < 30.0) &
        (df['duration'] >= 5.0) &
        (df['processed_text'].str.split().str.len() > 5)
    ]

    # XTTSv2 specified filters
    filtered_df = filtered_df[
        (filtered_df['duration'] < np.percentile(filtered_df['duration'].to_list(), 95)) &
        (filtered_df['processed_text'].str.len() < 166) &
        (filtered_df['processed_text'].str.contains('[a-zA-Z]', regex=True))
    ]

    filtered_df = filtered_df.loc[:, [
        "filepath", "duration", "processed_text"]]

    paths = filtered_df["filepath"]
    transcriptions = filtered_df["processed_text"]

    durations = [torchaudio.info(
        path).num_frames / torchaudio.info(path).sample_rate for path in paths]

    # Whisper's Feature Extractor will truncate samples more than 30 seconds during data loading which may lead to subpar results
    filtered_data = [(path, transcription, duration) for path, transcription, duration in zip(
        paths, transcriptions, durations) if duration < 30.0]
    total_duration = sum([sample[2] for sample in filtered_data])

    print(
        f"Total duration of {path} filtered train dataset of {len(filtered_data)} files: {total_duration / 60 / 60} hrs")

    return filtered_data


def load_arzen_llm(path: str) -> List[Tuple[str, str, float]]:

    if not os.path.exists(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    WAVS_PATH = os.path.join(path, 'wavs')
    TRANSCRIPTS_PATH = os.path.join(path, 'text')

    wav_files = os.listdir(WAVS_PATH)
    text_files = os.listdir(TRANSCRIPTS_PATH)

    wav_files.sort()
    text_files.sort()

    # Sanity checks
    if len(wav_files) != len(text_files):
        raise ValueError(
            "Number of WAV files and transcription files do not match!")

    for wav_path, transcript_path in zip(wav_files, text_files):
        if wav_path[:-4] != transcript_path[:-4]:
            raise ValueError("WAV file and transcription file do not match!")

    paths = []
    transcriptions = []
    durations = []

    # Loading data
    for wav_filename, text_filename in zip(wav_files, text_files):
        wav_path = os.path.join(WAVS_PATH, wav_filename)
        text_path = os.path.join(TRANSCRIPTS_PATH, text_filename)

        paths.append(wav_path)

        with open(text_path, 'r') as fin:
            transcriptions.append(fin.read().strip())

        info = torchaudio.info(wav_path)
        durations.append(info.num_frames / info.sample_rate)

    # Whisper's Feature Extractor will truncate samples more than 30 seconds during data loading which may lead to subpar results
    filtered_data = [(path, transcription, duration) for path, transcription, duration in zip(
        paths, transcriptions, durations) if duration < 30.0]
    total_duration = sum([sample[2] for sample in filtered_data])

    print(
        f"Total duration of {path} filtered train dataset of {len(filtered_data)} files: {total_duration / 60 / 60} hrs")

    return filtered_data


def load_egyptian_audio_books(path: str) -> List[Tuple[str, str, float]]:

    if not os.path.exists(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    WAVS_PATH = os.path.join(path, 'wavs')
    TRANSCRIPTS_PATH = os.path.join(path, 'metadata.csv')

    metadata = pd.read_csv(TRANSCRIPTS_PATH, sep='|', header=None)

    filenames = metadata[0].to_list()
    transcriptions = metadata[2].to_list()
    durations = []

    paths = []

    # Loading data
    for wav_filename in filenames:
        wav_path = os.path.join(WAVS_PATH, wav_filename + ".wav")
        paths.append(wav_path)

        info = torchaudio.info(wav_path)
        durations.append(info.num_frames / info.sample_rate)

    # Whisper's Feature Extractor will truncate samples more than 30 seconds during data loading which may lead to subpar results
    filtered_data = [(path, transcription, duration) for path, transcription, duration in zip(
        paths, transcriptions, durations) if duration < 30.0]
    total_duration = sum([sample[2] for sample in filtered_data])

    print(
        f"Total duration of {path} filtered train dataset of {len(filtered_data)} files: {total_duration / 60 / 60} hrs")

    return filtered_data


def load_synthetic_tts_data(wavs_path: str, metadata_file: str) -> List[Tuple[str, str, float]]:
    if not os.path.exists(wavs_path):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), wavs_path)

    if not os.path.exists(metadata_file):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), metadata_file)

    # Loading metadata file
    df = pd.read_csv(metadata_file, sep='\t')

    paths = [os.path.join(wavs_path, os.path.basename(path))
             for path in df['path'].to_list()]
    transcriptions = df['transcription'].to_list()
    durations = df['duration'].to_list()

    # Sanity checks
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), path)

    # Whisper's Feature Extractor will truncate samples more than 30 seconds during data loading which may lead to subpar results
    filtered_data = [(path, transcription, duration) for path, transcription, duration in zip(
        paths, transcriptions, durations) if duration < 30.0]
    total_duration = sum([sample[2] for sample in filtered_data])

    print(
        f"Total duration of {wavs_path} filtered train dataset of {len(filtered_data)} files: {total_duration / 60 / 60} hrs")

    return filtered_data


def load_mgb3(metadata_file: str) -> List[Tuple[str, str, float]]:
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), metadata_file)

    # Loading metadata file
    df = pd.read_csv(metadata_file, sep='\t')

    wavs_path = os.path.join(os.path.dirname(metadata_file), 'wavs')
    paths = [os.path.join(wavs_path, path) for path in df['path'].to_list()]

    transcriptions = df['transcription_ar'].to_list()
    durations = df['duration'].to_list()

    # Sanity checks
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), path)

    # Whisper's Feature Extractor will truncate samples more than 30 seconds during data loading which may lead to subpar results
    filtered_data = [(path, transcription, duration) for path, transcription, duration in zip(
        paths, transcriptions, durations) if duration < 30.0]
    total_duration = sum([sample[2] for sample in filtered_data])

    print(
        f"Total duration of {wavs_path} filtered train dataset of {len(filtered_data)} files: {total_duration / 60 / 60} hrs")

    return filtered_data


# Evaluation datasets
def load_fleurs_ar_eg_simple(split: str) -> List[Tuple[str, str]]:
    fleurs = load_dataset("google/fleurs", "ar_eg",
                          split="validation" if split == "dev" else split, trust_remote_code=True).to_pandas()
    df = fleurs.loc[:, ["path", "transcription"]]

    paths = df["path"]
    paths = [os.path.join(os.path.dirname(path), split, os.path.basename(
        path)) for path in paths]  # Fix some path error in fleurs dataset
    transcriptions = df["transcription"]

    return [(path, transcription) for path, transcription in zip(paths, transcriptions)]


def load_escwa(path: str) -> List[Tuple[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    DATA_PATH = os.path.join(path, "output")

    WAVS_PATHS = [os.path.join(DATA_PATH, file) for file in os.listdir(
        DATA_PATH) if file.endswith(".wav")]
    TRANSCRIPTS_PATHS = [os.path.join(DATA_PATH, file) for file in os.listdir(
        DATA_PATH) if file.endswith(".txt")]

    WAVS_PATHS.sort()
    TRANSCRIPTS_PATHS.sort()

    # Sanity checks
    if len(WAVS_PATHS) != len(TRANSCRIPTS_PATHS):
        raise ValueError(
            "Number of WAV files and transcription files do not match!")

    for wav_path, transcript_path in zip(WAVS_PATHS, TRANSCRIPTS_PATHS):
        if wav_path[:-4] != transcript_path[:-4]:
            raise ValueError("WAV file and transcription file do not match!")

    # Loading transcripts from transcription files
    TRANSCRIPTS = []
    for file in TRANSCRIPTS_PATHS:
        with open(file, "r") as fin:
            TRANSCRIPTS.append(fin.read().replace("<UNK>", "").strip())

    return [(path, transcription) for path, transcription in zip(WAVS_PATHS, TRANSCRIPTS)]


if __name__ == "__main__":

    # Test fine-tuning datasets loading functions

    dataset = load_fleurs_ar_eg(split="train")

    dataset = load_fleurs_ar_eg(split="dev")

    dataset = load_custom_yt_dataset(path="../data/processed/yt/metadata5.tsv")

    dataset = load_arzen_llm(path="../data/raw/arzen-llm")

    dataset = load_egyptian_audio_books(
        path="../data/raw/Egyptian-Audiobooks/output2")

    dataset = load_synthetic_tts_data(wavs_path="../data/processed/Synthetic-TTS-Data/wavs-cloned",
                                      metadata_file="../data/processed/Synthetic-TTS-Data/metadata-cloned-0.025-percent.tsv")

    dataset = load_mgb3(
        metadata_file="../data/interlim/MGB3/adapt.20170322.processed/metadata_nooverlap.tsv")
