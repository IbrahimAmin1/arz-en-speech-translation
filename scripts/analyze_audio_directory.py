"""
Analyze audio metadata from a directory of audio files

Example usage:
    $ python analyze_audio_directory.py --dirpath ../data/raw
"""

import os
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
import warnings

from tqdm import tqdm
import numpy as np
import torchaudio
from rich import print
from rich.pretty import pprint

warnings.filterwarnings("ignore", module="torchaudio") # Silence torchaudio warnings for MP3 because of soundfile backend


def find_audio_files(directory: str) -> list[str]:
    print(f"Getting all audio files from {directory}")

    audio_formats = {'.wav', '.mp3', '.flac', '.opus', '.aac', '.ogg'}
    files = []
    
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in audio_formats:
                files.append(os.path.join(root, filename))

    return files


def get_info(file_path: str) -> Optional[Tuple[str, Dict[str, float | int | str]]]:
    """
    Returns (file_path, info_dict) if successful, or (file_path, None) if broken
    """
    try:
        sample_info = torchaudio.info(file_path, backend="sox")
        
        # Get file size directly here to avoid redundant stat calls
        try:
            size_in_bytes = os.path.getsize(file_path)
            file_size = size_in_bytes / (1024 ** 3)  # Convert to GB directly
        except Exception:
            file_size = 0.0

        sampling_rate = sample_info.sample_rate
        num_samples = sample_info.num_frames
        audio_duration = num_samples / sampling_rate

        return (file_path, {
            "audio_duration": audio_duration,
            "sampling_rate": sampling_rate,
            "num_channels": sample_info.num_channels,
            "file_size": file_size,
            "bits_per_sample": sample_info.bits_per_sample,
            "encoding": sample_info.encoding,
        })

    except Exception:
        return (file_path, None)


def extract_audio_statistics(dir_path: str, audio_files: List[str]) -> Dict[str, Any]:
    if not audio_files:
        raise ValueError(f"No audio files detected in {dir_path}")

    with Pool(processes=cpu_count() // 2) as pool:
        results = list(tqdm(pool.imap(get_info, audio_files, chunksize=50), total=len(audio_files), desc='Generating metadata'))

    # Separate valid files from broken files
    valid_info = []
    broken_files = []
    
    for file_path, info in results:
        if info is None:
            broken_files.append(file_path)
        else:
            valid_info.append(info)

    if not valid_info:
        raise ValueError(f"No valid audio files could be processed in {dir_path}")

    # Extract arrays for statistics (only from valid files)
    durations = np.array([info["audio_duration"] for info in valid_info])
    sampling_rates = [info["sampling_rate"] for info in valid_info]
    num_channels = [info["num_channels"] for info in valid_info]
    file_sizes = [info["file_size"] for info in valid_info]
    bits_per_sample = [info["bits_per_sample"] for info in valid_info]
    encodings = [info["encoding"] for info in valid_info]

    # Sort durations once
    audio_files_durations = np.sort(durations)

    # Calculate statistics
    total_audio_files = len(audio_files_durations)
    total_duration_secs = float(np.sum(audio_files_durations))
    total_duration_hrs = total_duration_secs / 3600
    
    # Human readable duration
    hours = int(total_duration_secs // 3600)
    minutes = int((total_duration_secs % 3600) // 60)
    seconds = int(total_duration_secs % 60)
    human_readable_duration = f"{hours}h {minutes}m {seconds}s"

    # Compute statistics using numpy for efficiency
    mean_duration = float(np.mean(audio_files_durations))
    median_duration = float(np.median(audio_files_durations))
    max_duration = float(np.max(audio_files_durations))
    min_duration = float(np.min(audio_files_durations))

    # Get longest and shortest durations
    n_files = min(10, len(audio_files_durations))
    longest_n_duations = sorted(audio_files_durations[-n_files:].tolist(), reverse=True)
    shortest_n_durations = audio_files_durations[:n_files].tolist()

    # Calculate percentiles in one call
    percentiles = np.percentile(audio_files_durations, [25, 50, 75, 90, 95, 100])
    
    stats = {
        "total_audio_files": total_audio_files,
        "total_duration_secs": total_duration_secs,
        "total_duration_hrs": total_duration_hrs,
        "human_readable_duration": human_readable_duration,
        "mean_duration": mean_duration,
        "min_duration": min_duration,
        "median_duration": median_duration,
        "max_duration": max_duration,
        "percentile_25th": float(percentiles[0]),
        "percentile_50th": float(percentiles[1]),
        "percentile_75th": float(percentiles[2]),
        "percentile_90th": float(percentiles[3]),
        "percentile_95th": float(percentiles[4]),
        "percentile_100th": float(percentiles[5]),
        "longest_durations": longest_n_duations,
        "shortest_durations": shortest_n_durations,
        "unique_sampling_rates": sorted(list(set(sampling_rates))),
        "unique_num_channels": sorted(list(set(num_channels))),
        "unique_bits_per_sample": sorted(list(set(bits_per_sample))),
        "unique_encodings": sorted(list(set(str(e) for e in encodings))),
        "size": f"{sum(file_sizes)} GB",
        "broken_files": len(broken_files),
        "broken_files_list": broken_files
    }

    return stats


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Extract audio duration and other metadata from a directory of audio files')

    parser.add_argument('--dirpath', dest='dirpath', required=True,
                        type=str, help='Audio files directory path')

    args = parser.parse_args()

    DIR_PATH = args.dirpath

    if not os.path.exists(DIR_PATH):
        raise NotADirectoryError("Path doesn't exist.")

    all_audio_files = find_audio_files(directory=DIR_PATH)
    stats = extract_audio_statistics(dir_path=DIR_PATH, audio_files=all_audio_files)

    # Print to console
    pprint(stats, expand_all=True)
    
    # Save to JSON file in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "stats.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Statistics saved to: {output_file}")
