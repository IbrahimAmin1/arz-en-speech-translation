# Arz to En Speech-to-Text Translation <!-- omit in toc -->

[![License](<https://img.shields.io/badge/License-MIT-brightgreen.svg>)](https://opensource.org/licenses/MIT)
[![Python](<https://img.shields.io/badge/Python-3.10.19-blue>)](https://www.python.org/)
[![PyTorch](<https://img.shields.io/badge/PyTorch-2.6+cu124-FF5733>)](https://pytorch.org/)
[![Transformers](<https://img.shields.io/badge/Transformers-4.51.3-yellow>)](https://github.com/huggingface/transformers)
[![openai-whisper 20250625](https://img.shields.io/badge/openai--whisper-20250625-white.svg)](https://github.com/openai/whisper)

## Table of contents <!-- omit in toc -->

- [Publications \& Presentation](#publications--presentation)
- [Getting Started](#getting-started)
- [Paper Structure](#paper-structure)
- [Cascaded Models: Evaluation and Analysis](#cascaded-models-evaluation-and-analysis)
  - [ArzEn-ST: A Three-way Speech Translation Corpus for Code-Switched Egyptian Arabic - English](#arzen-st-a-three-way-speech-translation-corpus-for-code-switched-egyptian-arabic---english)
  - [ArzEn-LLM: Code-Switched Egyptian Arabic-English Translation and Speech Recognition Using LLMs](#arzen-llm-code-switched-egyptian-arabic-english-translation-and-speech-recognition-using-llms)
- [Proposed Methodology](#proposed-methodology)
- [Training and Evaluation Datasets](#training-and-evaluation-datasets)
  - [Training Datasets](#training-datasets)
  - [Evaluation Datasets](#evaluation-datasets)
- [Whisper Model Fine-tuning and Evaluation](#whisper-model-fine-tuning-and-evaluation)
- [Whisper Model Evaluations](#whisper-model-evaluations)
  - [Reported Zero-shot Whisper Paper Model Evaluations](#reported-zero-shot-whisper-paper-model-evaluations)
  - [Personal Zero-shot Whisper Model Evaluations](#personal-zero-shot-whisper-model-evaluations)
  - [Fine-tuned Whisper Model Evaluations](#fine-tuned-whisper-model-evaluations)
- [Environment, Requirements and Dependencies](#environment-requirements-and-dependencies)
- [References](#references)
  - [Research Paper and Presentation](#research-paper-and-presentation)
  - [Literature Survey](#literature-survey)
  - [Arabic Speech Datasets](#arabic-speech-datasets)
  - [LLM Leaderboards](#llm-leaderboards)
  - [Translation LLMs](#translation-llms)
  - [Useful Utils](#useful-utils)

## Publications & Presentation

For a deeper dive into our work, you can explore the full paper and the accompanying presentation:

- **Paper:** [ArzEn-E2E: Advancing End-to-End Speech Translation for Egyptian Arabic](https://www.overleaf.com/project/6929db0402a08f0749b23518)  
  A detailed study presenting the methodology, experiments, and results of our end-to-end speech translation model for Egyptian Arabic.

- **Presentation:** [ArzEn-E2E Presentation](https://docs.google.com/presentation/d/1-vVKJsEQ59G4NaCPfwJ7EiEfvRK2tsHbqS5wwKmGosw/edit?slide=id.p#slide=id.p)  
  A concise overview highlighting the key contributions, model architecture, and evaluation outcomes.

**Correspondence:**

Ibrahim Amin – *[IbrahimAmin532@gmail.com](mailto:IbrahimAmin532@gmail.com)* \
Dr. Fahima – *[fahima@aast.edu](mailto:fahima@aast.edu)*

## Getting Started

> Train an Arz-En speech-to-text translation (S2TT) model using LLM-translated, Egyptian Colloquial Arabic (ECA) code-switched, labeled speech datasets.

The goal is to produce a **high-quality S2TT model** suitable for:

1. Introducing the **first open-source, high-quality end-to-end (E2E) S2TT model** for the ECA dialect.
2. Delivering an **efficient model that performs well in low-resource environments**.

## Paper Structure

The paper is organized as follows:

1. **Abstract** – A concise summary of the problem, methodology, and key findings.
2. **Introduction** – Provides context, motivation, and objectives of the study.
3. **Literature Survey** – Reviews related work and includes comparative tables where relevant.
4. **Background** – Describes theoretical foundations and necessary concepts.
5. **Proposed Model** – Details the methodology, algorithms, and system design.
6. **Results** – Presents experimental outcomes, evaluations, and performance analysis.
7. **Discussion** – Interprets the results, highlights implications, and compares with existing approaches.
8. **Conclusion** – Summarizes contributions and key takeaways.
9. **Future Work** – Suggests directions for further research.
10. **References** – Lists all cited works in Springer-compliant format.

## Cascaded Models: Evaluation and Analysis

### ArzEn-ST: A Three-way Speech Translation Corpus for Code-Switched Egyptian Arabic - English

- TBC

### ArzEn-LLM: Code-Switched Egyptian Arabic-English Translation and Speech Recognition Using LLMs

- TBC

## Proposed Methodology

- TBC

## Training and Evaluation Datasets

### Training Datasets

- MGB-3
- DACS
- Arzen train ?

### Evaluation Datasets

- Fleurs
- ESCWA
- Arzen-ST

## Whisper Model Fine-tuning and Evaluation

```bash
cd src/

# Single-GPU Fine-tuning
python whisper_finetuning.py -m /path/to/pretrained-whisper-model -d {fleurs,custom-yt,arzen-llm,egyptian-audiobooks,mgb3,synthetic-data,mixed-data,all-real} -s /path/to/save/finetuned-whisper-model

# Whisper Model Evaluation
python whisper_evaluation.py --dataset-name {fleurs,escwa} --model-path /path/to/whisper-model
```

## Whisper Model Evaluations

### Reported Zero-shot Whisper Paper Model Evaluations

|      Model       | Parameters | Fleurs ar.eg Arabic WER% |
| :--------------: | :--------: | :----------------------: |
|   Whisper tiny   |    39 M    |          63.4%           |
|   Whisper base   |    74 M    |          48.8%           |
|  Whisper small   |   244 M    |          30.6%           |
|  Whisper medium  |   769 M    |          20.4%           |
|  Whisper large   |   1550 M   |          18.1%           |
| Whisper large-v2 |   1550 M   |          16.0%           |
| Whisper large-v3 |   1550 M   |           9.6%           |

<div style="page-break-before:always"></div>

### Personal Zero-shot Whisper Model Evaluations

|                               Model                               | Fleurs ar.eg Arabic `(transcription column)` WER% |
| :---------------------------------------------------------------: | :-----------------------------------------------: |
|   `Whisper base (fp32) beam search decoding with beam size = 5`   |            `48.5006%` **(45.2559%)***             |
|               Whisper small (fp32) greedy decoding                |                      32.40%                       |
|  `Whisper small (fp32) beam search decoding with beam size = 5`   |             `30.29%` **(26.1173%)***              |
|   Whisper small (fp16) beam search decoding with beam size = 5    |                     30.2780%                      |
|               Whisper medium (fp32) greedy decoding               |                     21.8768%                      |
|  `Whisper medium (fp32) beam search decoding with beam size = 5`  |            `20.3957%` **(15.2309%)***             |
|   Whisper medium (fp16) beam search decoding with beam size = 5   |                     20.4564%                      |
| `Whisper large-v2 (fp16) beam search decoding with beam size = 5` |                    `15.9766%`                     |
| `Whisper large-v3 (fp16) beam search decoding with beam size = 5` |             `14.4834%` **(9.5255%)***             |

*: basic_normalizer = BasicTextNormalizer(**remove_diacritics=True**)

### Fine-tuned Whisper Model Evaluations

| Whisper Model Variant | Fine-tuning dataset(s) | Fine-tuning Hyperparameters |         Inference Hyperparameters         | Fleurs ar.eg Arabic `(transcription column)` testset WER% | ESCWA WER% | Arzen ECA WER% | Arzen BLEU% |
| :-------------------: | :--------------------: | :-------------------------: | :---------------------------------------: | :-------------------------------------------------------: | :--------: | :------------: | :---------: |
|         base          |       Zero-shot        |              -              | (fp32) BSD, beam_size=5, chunk_length=30s |                          48.50%                           |  140.76%   |     XX.XX%     |   XX.XX%    |
|         small         |       Zero-shot        |              -              | (fp32) BSD, beam_size=5, chunk_length=30s |                          30.29%                           |   98.15%   |     XX.XX%     |   XX.XX%    |
|         base          | Translated MGB-3/DACS  |       (BS=16 & GA=2)*       | (fp32) BSD, beam_size=5, chunk_length=30s |                          XX.XX%                           |   XX.XX%   |     XX.XX%     |   XX.XX%    |
|         small         | Translated MGB-3/DACS  |       (BS=16 & GA=2)*       | (fp32) BSD, beam_size=5, chunk_length=30s |                          XX.XX%                           |   XX.XX%   |     XX.XX%     |   XX.XX%    |

<div style="page-break-before:always"></div>

`NOTES`:

- All **WER% scores** are calculated after **reference and hypothesis text normalization** using ```whisper.normalizers.BasicTextNormalizer```.
- *: All other Hyperparameters are unchanged in the python fine-tuning script `(src/whisper_finetuning.py)`.

## Environment, Requirements and Dependencies

- Ubuntu 24.04
- NVIDIA RTX 4060 Ti 16GB CUDA-Enabled GPU
- NVIDIA Drivers 570.195.03
- PyTorch 2.6.0 + CUDA Toolkit 12.4

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install ffmpeg cmake gcc libboost-dev build-essential gcc-multilib g++-multilib python3-dev zlib1g-dev

conda create -n arzen python=3.10 -y
conda activate arzen

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## References

### Research Paper and Presentation

- [ArzEn-E2E: Advancing End-to-End Speech Translation for Egyptian Arabic](https://www.overleaf.com/project/6929db0402a08f0749b23518)
- [ArzEn-E2E: Advancing End-to-End Speech Translation for Egyptian Arabic Presentation](https://docs.google.com/presentation/d/1-vVKJsEQ59G4NaCPfwJ7EiEfvRK2tsHbqS5wwKmGosw/edit?slide=id.p#slide=id.p)

### Literature Survey

- [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- [Speech Recognition Challenge in the Wild: Arabic MGB-3](https://arxiv.org/abs/1709.07276)
- [Effects of Dialectal Code-Switching on Speech Modules: A Study Using Egyptian Arabic Broadcast Speech](https://www.isca-archive.org/interspeech_2020/chowdhury20c_interspeech.html)
- [ArzEn: A Speech Corpus for Code-switched Egyptian Arabic-English](https://aclanthology.org/2020.lrec-1.523/)
- [ArzEn-ST: A Three-way Speech Translation Corpus for Code-Switched Egyptian Arabic - English](https://arxiv.org/abs/2211.12000)
- [ArzEn-LLM: Code-Switched Egyptian Arabic-English Translation and Speech Recognition Using LLMs](https://arxiv.org/abs/2406.18120)
- [ArzEn-MultiGenre: An aligned parallel dataset of Egyptian Arabic song lyrics, novels, and subtitles, with English translations](https://arxiv.org/abs/2508.01411)
- [FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech](https://arxiv.org/abs/2205.12446)
- [Arabic Code-Switching Speech Recognition using Monolingual Data](https://arxiv.org/abs/2107.01573)

### Arabic Speech Datasets

- [Mohamed Rashad Arabic Speech Datasets](https://huggingface.co/collections/MohamedRashad/arabic-speech-datasets)
- [QCRI Speech Corpus](https://huggingface.co/collections/ArabicSpeech/qcri-speech-corpus)
- [ARABIC NLP DATA CATALOGUE MASADER](https://arbml.github.io/masader/)

### LLM Leaderboards

- [silma-ai/Arabic-LLM-Broad-Leaderboard](https://huggingface.co/spaces/silma-ai/Arabic-LLM-Broad-Leaderboard)
- [OALL/Open-Arabic-LLM-Leaderboard](https://huggingface.co/spaces/OALL/Open-Arabic-LLM-Leaderboard)
- [Omartificial-Intelligence-Space/Arabic-MMMLU-Leaderborad](https://huggingface.co/spaces/Omartificial-Intelligence-Space/Arabic-MMMLU-Leaderborad)
- [Navid-AI/The-Arabic-RAG-Leaderboard](https://huggingface.co/spaces/Navid-AI/The-Arabic-RAG-Leaderboard)
- [MohamedRashad/arabic-tokenizers-leaderboard](https://huggingface.co/spaces/MohamedRashad/arabic-tokenizers-leaderboard)
- [LMArena](https://lmarena.ai/leaderboard)

### Translation LLMs

- [Command-A-Translate: Raising the Bar of Machine Translation with Difficulty Filtering](https://aclanthology.org/2025.wmt-1.55/)
- [Seed-X: Building Strong Multilingual Translation LLM with 7B Parameters](https://arxiv.org/abs/2507.13618)

### Useful Utils

- [Fix your Right-to-Left (RTL) text when you mix it with Left-to-Right (LTR) text](https://fixtxt.co/)
