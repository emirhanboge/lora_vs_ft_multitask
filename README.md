# Exploring LoRA’s Trade-Offs in Multi-Task Fine-Tuning for Representational Drift, Retention and Adaptation

This project investigates the effectiveness of Low-Rank Adaptation (LoRA) compared to full fine-tuning for multi-task learning, focusing on both similar (classification) and dissimilar (generation) tasks using LLaMA models.

## Project Overview

The project consists of four main components:
1. Multi-task training framework
2. Performance analysis tools
3. SVD (Singular Value Decomposition) analysis
4. Dataset preprocessing utilities

### Key Features

- Support for both similar tasks (classification) and dissimilar tasks (generation)
- Implementation of LoRA with configurable ranks (4, 8, 16, 32, 64)
- Comprehensive performance analysis across training checkpoints
- SVD analysis for understanding representational drift
- Support for multiple evaluation metrics

## Project Structure

```
.
├── config.py                    # Configuration settings and constants
├── data_preprocessing.py        # Dataset preprocessing utilities
├── multi_task_training.py      # Main training script
├── eval.py                     # Evaluation utilities
├── retention_eval.py           # Retention evaluation script
├── svd_analysis.py            # SVD analysis of model weights
├── checkpoint_utils.py         # Checkpoint management utilities
├── dataset_descriptions.py     # Dataset documentation
├── training_configs.py         # Training configurations
├── plot_results.ipynb         # Jupyter notebook for plotting results
├── requirements.txt           # Project dependencies
├── .env.example              # Environment variables template
└── directories:
    ├── data/                 # Dataset storage
    ├── models/              # Model checkpoints
    ├── logs/               # Training logs
    ├── outputs/           # Output files and results
    ├── tokenizer/        # Tokenizer files
    ├── drift_analysis_plots/  # SVD analysis plots
    └── performance_analysis/  # Performance analysis results
```

## Tasks and Metrics

### Similar Tasks (Classification)
- **SST2** (Sentiment Analysis)
  - Metrics: Accuracy, F1-score
- **MNLI** (Natural Language Inference)
  - Metrics: Accuracy, F1-score
- **QQP** (Question Pair Classification)
  - Metrics: Accuracy, F1-score

### Dissimilar Tasks (Generation)
- **SQuAD v2** (Question Answering)
  - Metrics: F1-score
- **CodeXGLUE** (Code Generation)
  - Metrics: BLEU score
- **CNN-DailyMail** (Text Summarization)
  - Metrics: ROUGE-L score

## Analysis Components

### Performance Analysis
- Evaluation on both validation and test splits
- Comparison across different LoRA ranks
- Tracking of multiple metrics per task
- Performance visualization and plotting

### SVD Analysis
- Analysis of weight matrix structure
- Tracking of representational drift from pretrained model
- Comparison of singular value distributions
- Analysis of Q and V projection layers

## Setup and Installation

1. Install dependencies:
```bash
pip install torch transformers datasets evaluate peft numpy matplotlib seaborn safetensors
```

2. Set up environment variables:
```bash
export DATA_DIR="path/to/data"
export OUTPUT_DIR="path/to/output"
```

## Usage

### Training

1. Train on similar tasks:
```bash
python multi_task_training.py --similar --lora 8  # Use LoRA with rank 8
python multi_task_training.py --similar --full-finetune  # Use full fine-tuning
```

2. Train on dissimilar tasks:
```bash
python multi_task_training.py --dissimilar --lora 8
python multi_task_training.py --dissimilar --full-finetune
```

### Analysis

1. Performance Analysis:
```bash
python eval.py
```

2. SVD Analysis:
```bash
python svd_analysis.py
```

The analysis will:
- Evaluate models on validation and test splits
- Generate performance plots
- Analyze representational drift
- Save results to JSON format

## Model Configurations

- Base Model: LLaMA 1B
- LoRA Configurations:
  - Ranks: 4, 8, 16, 32, 64
  - Target modules: Q and V projections
  - Alpha scaling: 2x rank
- Full fine-tuning baseline

## Acknowledgments

- LLaMA model from Meta AI
- Hugging Face Transformers library
- PEFT library from Hugging Face

## License
This project is licensed under the terms of the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
