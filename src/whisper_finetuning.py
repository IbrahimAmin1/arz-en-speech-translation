import argparse
import os
from typing import List, Tuple, Any, Dict, Union
import warnings
import math
from dataclasses import dataclass

from whisper_dataset_utils import load_fleurs_ar_eg, load_custom_yt_dataset, load_arzen_llm, load_egyptian_audio_books, load_mgb3, load_synthetic_tts_data

import torchaudio
import transformers
from torch.utils.data import Dataset
import torch
from transformers import enable_full_determinism
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
import evaluate


warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_warning()

SEED = 42
enable_full_determinism(seed=SEED, warn_only=False)


class WhisperDataset(Dataset):
    def __init__(self, dataset_split: List[Tuple[str, str, float]]):
        self.data = dataset_split
        self.SAMPLE_RATE = 16_000
        self.CHANNELS = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        signal, sr = torchaudio.load(self.data[index][0])
        transcript = self.data[index][1]

        # Resample the audio (if not already)
        if sr != self.SAMPLE_RATE:
            resample_transform = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.SAMPLE_RATE)
            signal = resample_transform(signal)

        # Convert to monochannel (if not already)
        if signal.shape[0] != self.CHANNELS:
            signal = torch.mean(signal, dim=0, keepdim=True)

        # Prepare Mel Spectrogram and tokenize transcript
        input_features = feature_extractor(signal.squeeze(
        ), sampling_rate=self.SAMPLE_RATE, return_tensors="pt").input_features[0]
        input_ids = tokenizer(transcript).input_ids

        features = {"input_features": input_features, "labels": input_ids}

        return features


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        # Note that no additional padding is applied here since the inputs are of fixed dimension, the input_features are simply converted to PyTorch tensors.
        input_features = [{"input_features": feature["input_features"]}
                          for feature in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper model on a custom dataset")

    parser.add_argument("--model", "-m", required=True, type=str,
                        help="Local whisper model path or 🤗 Hub Whisper model ID")
    parser.add_argument("--dataset", "-d", required=True, type=str,  choices=[
                        "fleurs", "custom-yt", "arzen-llm", "egyptian-audiobooks", "mgb3", "synthetic-data", "mixed-data", "all-real"], help="Fine-tuning dataset")
    parser.add_argument("--save", "-s", required=True,
                        type=str, help="Fine-tuned model save path")
    args = parser.parse_args()

    ####

    # Define some training hyperparameters
    MODEL_VARIANT = args.model
    DATASET = args.dataset
    SAVE_PATH = args.save

    NUM_EPOCHS = 10
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 2
    LEARNING_RATE = 1e-5
    WARMUP_STEPS = 100
    LOGGING_STEPS = 200

    # Load fine-tuning dataset
    if DATASET == "fleurs":
        # Load train split of Fleurs ar.eg dataset
        SPLIT = "train"
        dataset_train = load_fleurs_ar_eg(split=SPLIT, threshold=math.inf)
    elif DATASET == "custom-yt":
        # Load train Custom YT dataset
        PATH = "../data/processed/yt/metadata5.tsv"
        dataset_train = load_custom_yt_dataset(path=PATH)
    elif DATASET == "arzen-llm":
        # Load train arzen-llm dataset
        PATH = "../data/raw/arzen-llm"
        dataset_train = load_arzen_llm(path=PATH)
    elif DATASET == "egyptian-audiobooks":
        # Load train egyptian audiobooks dataset
        PATH = "../data/raw/Egyptian-Audiobooks/output2"
        dataset_train = load_egyptian_audio_books(path=PATH)
    elif DATASET == "mgb3":
        # Load MGB-3 dataset
        PATH = "../data/interlim/MGB3/adapt.20170322.processed/metadata_nooverlap.tsv"
        dataset_train = load_mgb3(metadata_file=PATH)
    elif DATASET == "synthetic-data":
        # Load train synthetic dataset
        WAVS_PATH = "../data/processed/Synthetic-TTS-Data/wavs-cloned"
        METADATA_PATH = "../data/processed/Synthetic-TTS-Data/metadata-0.025-percent.tsv"
        dataset_train = load_synthetic_tts_data(
            wavs_path=WAVS_PATH, metadata_file=METADATA_PATH)
    elif DATASET == "mixed-data":
        # Load and combine synthetic data and real data for training
        FLEURS_SPLIT = "train"
        ARZEN_PATH = "../data/raw/arzen-llm"
        SYNTHETIC_WAVS_PATH = "../data/processed/Synthetic-TTS-Data/wavs-cloned"
        SYNTHETIC_METADATA_PATH = "../data/processed/Synthetic-TTS-Data/metadata-cloned-0.025-percent.tsv"
        dataset_train = load_fleurs_ar_eg(split=FLEURS_SPLIT) + load_synthetic_tts_data(
            wavs_path=SYNTHETIC_WAVS_PATH, metadata_file=SYNTHETIC_METADATA_PATH) + load_arzen_llm(path=ARZEN_PATH)
        print(
            f"Total duration of all training datasets of {len(dataset_train)} Files: {sum([sample[2] for sample in dataset_train]) / 60 / 60} hrs")
    elif DATASET == "all-real":
        # Load and combine all previous "real" train datasets
        FLEURS_SPLIT = "train"
        YT_PATH = "../data/processed/yt/metadata5.tsv"
        ARZEN_PATH = "../data/raw/arzen-llm"
        EGYPTIAN_AUDIOBOOKS_PATH = "../data/raw/Egyptian-Audiobooks/output2"
        MGB3_ADAPT_PATH = "../data/interlim/MGB3/adapt.20170322.processed/metadata_nooverlap.tsv"
        dataset_train = load_fleurs_ar_eg(split=FLEURS_SPLIT) + load_custom_yt_dataset(path=YT_PATH) + load_arzen_llm(
            path=ARZEN_PATH) + load_egyptian_audio_books(path=EGYPTIAN_AUDIOBOOKS_PATH) + load_mgb3(metadata_file=MGB3_ADAPT_PATH)
        print(
            f"Total duration of all training datasets of {len(dataset_train)} Files: {sum([sample[2] for sample in dataset_train]) / 60 / 60} hrs")
    else:
        raise ValueError(f"Invalid dataset: {DATASET}")

    # Load validation split of Fleurs ar.eg dataset
    dataset_val = load_fleurs_ar_eg(split="dev")

    print('\n')

    ####

    # Load Whisper feature extractor and tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        pretrained_model_name_or_path=MODEL_VARIANT)
    tokenizer = WhisperTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_VARIANT, language="arabic", task="transcribe")
    processor = WhisperProcessor(
        feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Prepare train and val datasets
    dataset_train = WhisperDataset(dataset_split=dataset_train)
    dataset_val = WhisperDataset(dataset_split=dataset_val)

    # Load Whisper Model and Initialize config
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_VARIANT)

    model.config.use_cache = False  # For gradient checkpointing
    model.generation_config.language = "arabic"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # Initialize data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Initialize Evaluation metrics
    metric = evaluate.load("wer")

    # Define callback for early stopping
    callback = EarlyStoppingCallback(early_stopping_patience=3)

    # Initialize Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(SAVE_PATH, 'checkpoints'),
        logging_dir=os.path.join(SAVE_PATH, 'runs'),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        group_by_length=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        report_to=["tensorboard"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': True},
        fp16=True,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=1,
        disable_tqdm=False,
        seed=SEED,
        data_seed=SEED,
    )

    # Start training !!!
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[callback],
    )

    trainer.train()

    # Saving best Checkpoint
    trainer.save_model(os.path.join(SAVE_PATH, "best_model"))
    processor.save_pretrained(os.path.join(SAVE_PATH, "best_model"))
