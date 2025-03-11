# Multi-Task Learning Analysis: LoRA vs Full Fine-tuning

This repository provides comprehensive analysis tools for comparing LoRA (Low-Rank Adaptation) and full fine-tuning approaches in multi-task learning settings using LLaMA-2-1B. The analysis examines both similar tasks (classification) and dissimilar tasks (generation) through singular value decomposition and gradient flow analysis.

## Overview

### Analysis Components
1. **Singular Value Decomposition (SVD) Analysis**
   - Analyzes weight matrix structure and changes
   - Compares model spaces with pre-trained model
   - Tracks rank evolution during training

2. **Gradient Flow Analysis**
   - Examines gradient properties across layers
   - Analyzes training dynamics
   - Studies impact of LoRA rank on optimization

### Model Configurations
- **Base Model**: LLaMA-2-1B
- **Training Approaches**:
  - Full fine-tuning (complete parameter update)
  - LoRA with ranks: 4, 8, 16, 32, 64
- **Checkpoints**: Multiple snapshots during training

### Task Types

#### Similar Tasks (Classification)
- **SST2**: Binary sentiment analysis
- **MNLI**: Natural language inference (3-way classification)
- **QQP**: Question pair similarity detection

#### Dissimilar Tasks (Generation)
- **Question Answering**: Open-domain QA
- **Code Generation**: Python code synthesis
- **Text Summarization**: Abstractive summarization

## Technical Details

### Singular Value Analysis (`singular_value_analysis.py`)

#### Weight Matrix Analysis
```python
def get_weight_matrices(model):
    """Extracts matrices from:
    - Attention layers (Q, K, V, O projections)
    - Classification head (similar tasks)
    - LM head (dissimilar tasks)
    """
```

#### SVD Metrics
1. **Basic Metrics**
   - Singular values and vectors (U, S, Vh)
   - Effective rank: \( \sum_{i} (s_i > 10^{-5}) \)
   - Spectral norm: \( \max(s_i) \)
   - Condition number: \( s_{\max}/s_{\min} \)
   - 95% energy rank: \( \arg\min_k (\sum_{i=1}^k s_i / \sum_{i} s_i \geq 0.95) \)

2. **Comparison with Pre-trained Model**
   - Singular value correlation (Spearman)
   - Left/right space similarities (cosine)
   - Principal angles between subspaces
   ```python
   # Principal angles computation
   k = min(10, U.shape[1], U_pre.shape[1])
   Q1, Q2 = U[:, :k], U_pre[:, :k]
   angles = np.arccos(np.clip(np.linalg.svd(Q1.T @ Q2)[1], -1.0, 1.0))
   ```

### Gradient Analysis (`gradient_analysis.py`)

#### Sample Input Generation
```python
def get_sample_inputs(task_type):
    """
    Similar tasks: "This movie is great!"
    Dissimilar tasks: "Question: What is the capital of France? Answer:"
    Max length: 128 tokens
    """
```

#### Gradient Statistics
1. **Global Metrics**
   - Mean absolute gradient
   - Standard deviation
   - L2 norm
   - Sparsity (fraction of zeros)

2. **Layer-wise Analysis**
   ```python
   layer_types = ["attention", "mlp", "embed", "norm", "head"]
   metrics = {
       "mean": np.mean(np.abs(grad)),
       "std": np.std(grad),
       "norm": np.linalg.norm(grad),
       "sparsity": np.mean(grad == 0)
   }
   ```

## Implementation Architecture

### Model Loading System
```python
def load_model(config, step=None):
    """
    Similar tasks: AutoModelForSequenceClassification (3 labels)
    Dissimilar tasks: AutoModelForCausalLM
    Handles: Pretrained, LoRA, Full fine-tuning
    """
```

### Analysis Pipeline

1. **Data Collection**
   ```python
   checkpoint_steps = range(
       config["num_checkpoints"],
       config["num_checkpoints"] * 1000 + 1,
       1000
   )
   ```

2. **Memory Management**
   - GPU memory cleanup after each analysis
   - Proper model unloading
   - Gradient cleanup between computations

3. **Results Storage**
   - CSV format for raw data
   - Organized visualization directory structure
   - Task-specific subdirectories

## Visualization Suite

### SVD Analysis Visualizations

1. **Singular Value Decay**
   - Log-scale plots
   - Model-wise comparison
   - First 10 singular values

2. **Rank Evolution**
   - Effective rank over training
   - 95% energy rank changes
   - Layer-wise comparisons

3. **Model Space Analysis**
   - Principal angle evolution
   - Similarity with pretrained model
   - Layer-wise space comparisons

### Gradient Analysis Visualizations

1. **Global Statistics**
   - Training dynamics
   - Rank impact analysis
   - Layer-type comparisons

2. **Layer-wise Analysis**
   - Gradient distribution by layer
   - Sparsity patterns
   - Norm evolution

## Output Structure

```
project_root/
├── svd_analysis/
│   ├── similar/
│   │   ├── singular_value_decay.png
│   │   ├── effective_rank_comparison.png
│   │   ├── pretrained_similarity.png
│   │   ├── training_dynamics.png
│   │   └── lora_rank_analysis.png
│   ├── dissimilar/
│   │   └── [same structure as similar]
│   └── svd_analysis_results.csv
│
└── gradient_analysis/
    ├── similar/
    │   ├── global_gradient_stats.png
    │   ├── layer_wise_*.png
    │   ├── lora_rank_analysis.png
    │   └── gradient_evolution.png
    ├── dissimilar/
    │   └── [same structure as similar]
    └── gradient_analysis_results.csv
```

## Usage

### Command Line Interface
```bash
# SVD Analysis
python singular_value_analysis.py --task-type [similar|dissimilar|both]

# Gradient Analysis
python gradient_analysis.py --task-type [similar|dissimilar|both]
```

### Configuration Format
```python
MODEL_CONFIG = {
    "name": "model_path",
    "num_checkpoints": int,
    "type": ["full_ft"|"lora"],
    "rank": int,  # for LoRA only
    "task_type": ["similar"|"dissimilar"]
}
```

## Dependencies

### Core Libraries
- PyTorch (>=1.10.0)
- Transformers (>=4.20.0)
- PEFT (>=0.3.0)

### Analysis Tools
- NumPy (>=1.20.0)
- Pandas (>=1.3.0)
- SciPy (>=1.7.0)

### Visualization
- Matplotlib (>=3.4.0)
- Seaborn (>=0.11.0)

## Multi-task Training

### Training Script (`multi_task_training.py`)

The training script handles both similar (classification) and dissimilar (generation) tasks:

```python
def run_training_configuration(config: TrainingConfig):
    """
    Handles:
    - Similar tasks: SST2, MNLI, QQP (classification)
    - Dissimilar tasks: QA, Code Generation, Summarization
    - Both LoRA and full fine-tuning approaches
    """
```

### Usage

```bash
# For similar tasks (classification)
python multi_task_training.py --task-type similar --config-index [0-5]

# For dissimilar tasks (generation)
python multi_task_training.py --task-type dissimilar --config-index [0-5]
```

Where `config-index` represents:
- `0`: Full fine-tuning
- `1`: LoRA (rank 4)
- `2`: LoRA (rank 8)
- `3`: LoRA (rank 16)
- `4`: LoRA (rank 32)
- `5`: LoRA (rank 64)

### Training Configurations

```python
class TrainingConfig:
    model_name: str          # Base LLaMA model path
    batch_size: int         # Per device batch size
    num_epochs: int         # Number of training epochs
    learning_rate: float    # Initial learning rate
    warmup_ratio: float    # Portion of training for LR warmup
    weight_decay: float    # Weight decay for AdamW
    use_lora: bool         # Whether to use LoRA
    lora_rank: int         # LoRA rank (if use_lora=True)
    lora_alpha: int        # LoRA alpha parameter
    lora_dropout: float    # LoRA dropout probability
    task_type: str         # "similar" or "dissimilar"
```

### Training Features

1. **Data Loading**
   - Balanced sampling across tasks
   - Task-specific preprocessing
   - Proper tokenization with max length 1024

2. **Model Configuration**
   - Automatic model type selection based on task
   - LoRA adaptation for efficient training
   - Proper weight initialization

3. **Training Process**
   - Gradient accumulation for large batches
   - Mixed precision training (bfloat16)
   - Gradient checkpointing for memory efficiency
   - Proper checkpoint saving

4. **Task-specific Features**

   Similar Tasks:
   ```python
   class SimilarTaskTrainer:
       """Classification task trainer with:
       - Shared classification head
       - Task-specific metrics
       - GLUE evaluation
       """
   ```

   Dissimilar Tasks:
   ```python
   class DissimilarTaskTrainer:
       """Generation task trainer with:
       - Task-specific prompts
       - Multiple evaluation metrics
       - Generation-specific parameters
       """
   ```

### Output Structure

```
outputs/
├── configs/
│   └── {model_name}.json       # Training configuration
├── checkpoints/
│   ├── checkpoint-{step}/      # Model checkpoints
│   └── final/                  # Final model
└── logs/
    ├── training_log.txt        # Training progress
    └── tensorboard/            # TensorBoard logs
```

### Memory Requirements

- Similar tasks: ~16GB GPU RAM
- Dissimilar tasks: ~24GB GPU RAM
- Full fine-tuning: Additional ~4GB
- Gradient checkpointing reduces memory by ~40%

## Notes and Best Practices

### Memory Management
- Use `torch.cuda.empty_cache()` after model analysis
- Delete unused model objects
- Monitor GPU memory usage

### Analysis Tips
- Run similar and dissimilar tasks separately for better memory management
- Use smaller batch sizes for gradient analysis
- Consider layer-wise analysis for detailed insights

### Visualization Guidelines
- Use consistent color schemes for comparison
- Include error bars where applicable
- Save high-resolution figures for publication

## Citation

If you use this analysis in your research, please cite:

```bibtex
@misc{lora_multitask_analysis,
  title={Comparative Analysis of LoRA and Full Fine-tuning in Multi-task Learning},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/repo}}
}
```

# Multi-Task Training for LLaMA Models

This repository contains code for training LLaMA models on multiple tasks simultaneously, using either full fine-tuning or LoRA (Low-Rank Adaptation).

## Setup

1. Install the required dependencies:
```bash
pip install transformers datasets peft evaluate huggingface_hub
```

2. Set up your Hugging Face credentials (optional, for uploading datasets and models):
```bash
export HF_USERNAME="your-username"
export HF_TOKEN="your-token"
```

## Running the Script

The `multi_task_training.py` script supports various command-line arguments to control the training process:

### Preparing Datasets Only

To only prepare and upload the datasets without training:

```bash
python multi_task_training.py --prepare-datasets-only
```

This will process the datasets for the dissimilar tasks (default) and upload them to the Hugging Face Hub if credentials are provided.

### Training with Different LoRA Ranks

To train with a specific LoRA rank:

```bash
python multi_task_training.py --lora-rank 16
```

This will train a model with LoRA rank 16 on the dissimilar tasks.

### Training with Full Fine-Tuning

To train with full fine-tuning instead of LoRA:

```bash
python multi_task_training.py --full-ft
```

### Training on Different Task Types

To train on similar tasks (classification):

```bash
python multi_task_training.py --task-type similar
```

To train on dissimilar tasks (generation):

```bash
python multi_task_training.py --task-type dissimilar
```

To train on all tasks:

```bash
python multi_task_training.py --task-type all
```

### Uploading Models to Hugging Face Hub

To upload the trained models to the Hugging Face Hub:

```bash
python multi_task_training.py --upload
```

### Forcing Dataset Upload

To force re-upload of datasets even if they already exist on the Hub:

```bash
python multi_task_training.py --prepare-datasets-only --force-upload
```

### Complete Example

To prepare datasets, train a model with LoRA rank 16 on dissimilar tasks, and upload the model to the Hub:

```bash
python multi_task_training.py --task-type dissimilar --lora-rank 16 --upload
```

## Dataset Structure

The script processes and combines the following datasets:

### Dissimilar Tasks
- SQuAD v2 (Question Answering)
- Code-to-Text (Code Summarization)
- CNN/DailyMail (Text Summarization)

### Similar Tasks
- SST-2 (Sentiment Analysis)
- MNLI (Natural Language Inference)
- QQP (Question Pair Similarity)

## Model Configurations

The script supports various training configurations:

- Full fine-tuning
- LoRA with ranks: 4, 8, 16, 32, 64

Each configuration can be applied to similar tasks, dissimilar tasks, or both.