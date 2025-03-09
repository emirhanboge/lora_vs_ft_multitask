# TreeMark: Tree-Structured Task Benchmark for LLMs

TreeMark is a benchmark suite for evaluating Large Language Models (LLMs) on a tree-structured collection of tasks, including both similar tasks (classification) and dissimilar tasks (generation).

## Pre-trained Models and Datasets

All pre-trained models and preprocessed datasets are available on the Hugging Face Hub:

### Pre-trained Models

#### Single-Task Models
- SST2: [Full Fine-tuning](https://huggingface.co/emirhanboge/LLaMA_1B_sst2_FullFT) | [LoRA-16](https://huggingface.co/emirhanboge/LLaMA_1B_sst2_LoRA_16)
- MNLI: [Full Fine-tuning](https://huggingface.co/emirhanboge/LLaMA_1B_mnli_FullFT) | [LoRA-16](https://huggingface.co/emirhanboge/LLaMA_1B_mnli_LoRA_16)
- QQP: [Full Fine-tuning](https://huggingface.co/emirhanboge/LLaMA_1B_qqp_FullFT) | [LoRA-16](https://huggingface.co/emirhanboge/LLaMA_1B_qqp_LoRA_16)

#### Multi-Task Models (Similar Tasks)
- [Full Fine-tuning](https://huggingface.co/emirhanboge/LLaMA_1B_sst2_mnli_qqp_FullFT_)
- LoRA Models:
  - [LoRA-4](https://huggingface.co/emirhanboge/LLaMA_1B_sst2_mnli_qqp_LoRA_4)
  - [LoRA-8](https://huggingface.co/emirhanboge/LLaMA_1B_sst2_mnli_qqp_LoRA_8)
  - [LoRA-16](https://huggingface.co/emirhanboge/LLaMA_1B_sst2_mnli_qqp_LoRA_16)
  - [LoRA-32](https://huggingface.co/emirhanboge/LLaMA_1B_sst2_mnli_qqp_LoRA_32)
  - [LoRA-64](https://huggingface.co/emirhanboge/LLaMA_1B_sst2_mnli_qqp_LoRA_64)

### Preprocessed Datasets

#### Similar Tasks (Classification)
- [SST2](https://huggingface.co/datasets/emirhanboge/sst2_llama1b_modified)
- [MNLI](https://huggingface.co/datasets/emirhanboge/mnli_llama1b_modified)
- [QQP](https://huggingface.co/datasets/emirhanboge/qqp_llama1b_modified)
- [Combined (SST2+MNLI+QQP)](https://huggingface.co/datasets/emirhanboge/sst2_mnli_qqp_llama1b_modified)

#### Dissimilar Tasks (Generation)
- [SQuAD v2](https://huggingface.co/datasets/emirhanboge/rajpurkar_squad_v2_llama1b_modified)
- [CNN/DailyMail](https://huggingface.co/datasets/emirhanboge/cnn_dailymail_llama1b_modified)
- [CodeXGLUE](https://huggingface.co/datasets/emirhanboge/codex_glue_llama1b_modified)
- [Combined (QA+Code+Summarization)](https://huggingface.co/datasets/emirhanboge/qa_code_summarization_llama1b_modified)

## Project Structure

```
.
├── config.py                   # Global configuration variables
├── training_configs.py         # Training configurations and hyperparameters
├── data_preprocessing.py       # Dataset preprocessing and tokenization
├── single_task_training.py     # Single-task training script
├── multi_task_training.py      # Multi-task training script
└── evaluate_and_upload.py      # Model evaluation and HuggingFace Hub upload
```

## Configuration System

The project uses a centralized configuration system with two main components:

1. `config.py`: Global constants and paths
2. `training_configs.py`: Training configurations for different experiments

### Training Configurations

We provide predefined configurations for both similar and dissimilar tasks:

- **Similar Tasks** (Classification: SST2, MNLI, QQP)
  - Full fine-tuning
  - LoRA with ranks: 4, 8, 16 (baseline), 32, 64

- **Dissimilar Tasks** (Generation: QA, Code, Summarization)
  - Full fine-tuning
  - LoRA with ranks: 4, 8, 16 (baseline), 32, 64

Each configuration specifies:
- Model parameters (LoRA settings, target modules)
- Training parameters (batch size, epochs, learning rate)
- Saving and logging settings

## Usage

### Single-Task Training

Train a model on a single classification task:

```bash
python single_task_training.py --task TASK --config-index INDEX

# Examples:
python single_task_training.py --task sst2 --config-index 3  # LoRA rank 16
python single_task_training.py --task mnli --config-index 0  # Full fine-tuning
```

Available tasks: `sst2`, `mnli`, `qqp`

### Multi-Task Training

Train a model on multiple tasks simultaneously:

```bash
python multi_task_training.py --task-type TYPE --config-index INDEX

# Examples:
python multi_task_training.py --task-type similar --config-index 3      # Classification tasks, LoRA rank 16
python multi_task_training.py --task-type dissimilar --config-index 0   # Generation tasks, full fine-tuning
```

Task types: `similar` (classification), `dissimilar` (generation)

### Configuration Index Reference

The `config-index` parameter maps to different training configurations:
- `0`: Full fine-tuning
- `1`: LoRA rank 4
- `2`: LoRA rank 8
- `3`: LoRA rank 16 (original baseline)
- `4`: LoRA rank 32
- `5`: LoRA rank 64

### Model Output Structure

Models are saved with consistent naming conventions:
- Single-task models: `single_TASK_CONFIG`
- Multi-task models: `TYPE_CONFIG`

Where:
- `TASK`: Task name (e.g., `sst2`, `mnli`, `qqp`)
- `TYPE`: Task type (`similar` or `dissimilar`)
- `CONFIG`: Configuration type (`FullFT` for full fine-tuning or `LoRA_R` for LoRA with rank R)

Example model names:
- `single_sst2_similar_LoRA_16`
- `similar_LoRA_32`
- `dissimilar_FullFT`

### Configuration Files

Training configurations are automatically saved in the `configs` directory for reproducibility:
```
configs/
├── single_sst2_similar_LoRA_16.json
├── similar_LoRA_32.json
└── dissimilar_FullFT.json
```

## Training Parameters

Default parameters for different setups:

### Similar Tasks (Classification)
- Full fine-tuning:
  - Batch size: 128
  - Learning rate: 1e-5
  - Weight decay: 0
  - Gradient accumulation: 4

- LoRA:
  - Batch size: 32
  - Learning rate: 1e-5
  - Weight decay: 0.01
  - Gradient accumulation: 4
  - LoRA dropout: 0.1
  - Target modules: ["q_proj", "v_proj"]

### Dissimilar Tasks (Generation)
- Full fine-tuning and LoRA:
  - Batch size: 32
  - Learning rate: 1e-5
  - Weight decay: 0.01
  - Gradient accumulation: 4
  - LoRA settings: Same as classification

## Evaluation Metrics

- **Classification Tasks**:
  - Accuracy
  - F1 Score
  - GLUE metrics

- **Generation Tasks**:
  - SQUAD metrics (QA)
  - ROUGE scores (Summarization)
  - BLEU score (Code Generation)

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- PEFT
- Datasets
- Evaluate
- tqdm

## License

[Insert License Information]