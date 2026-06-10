"""
Generate a processed training dataset

Example usage:
    $ python processed_dataset_generator.py --dirpath ../data/raw
"""

import os
import argparse

import pandas as pd


filepaths = []
durations = []
arabic_texts = []
normalized_texts = []
csw_texts = []
llm_translated_texts = []
splits = []
sources = []



def translate_mgb3(dirname: str):
    
    for split in ["train", "dev", "test"]:
        
        split_wavs_path = os.path.join(dirname, split)    


def translate_dacs(dirname: str):
    
    metadata_file = os.path.join(dirname, "sentences.tsv")


def process_arzen(dirname: str):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate processed training dataset')
    parser.add_argument('--dirpath', type=str, default='../data/raw',
                        help='Path to raw data directory (default: ../data/raw)')
    parser.add_argument('--output', type=str, default='../data/processed',
                        help='Path to save processed data (default: ../data/processed)')
    
    args = parser.parse_args()
    
    DATA_PATH = args.dirpath
    SAVE_PATH = args.output
    
    if not os.path.exists(DATA_PATH):
        raise ValueError(f"Data path does not exist: {DATA_PATH}")
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        print(f"Created output directory: {SAVE_PATH}")
    
    # Process datasets
    print("Processing datasets...")
    translate_mgb3(os.path.join(DATA_PATH, 'mgb3_ar'))
    translate_dacs(os.path.join(DATA_PATH, 'DACS'))
    process_arzen(os.path.join(DATA_PATH, 'ArzEn'))
    
    print("Processing complete!")
