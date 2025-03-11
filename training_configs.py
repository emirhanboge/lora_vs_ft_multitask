"""
Training configurations for TreeMark.

This module contains all training configurations for both similar and dissimilar tasks,
including different LoRA ranks and full fine-tuning settings.
"""

from dataclasses import dataclass
from typing import List, Optional, Literal

TaskType = Literal["similar", "dissimilar"]

@dataclass
class TrainingConfig:
    """Configuration for training runs."""
    # Task type
    task_type: TaskType = "similar"
    tasks: List[str] = None  # List of tasks to train on
    
    # Model configuration
    base_model: str = "meta-llama/Llama-3.2-1B"  # Base LLaMA model to use
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: Optional[int] = None  # If None, will be set to lora_rank * 2
    lora_dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4
    
    # Evaluation parameters
    eval_steps: int = 500
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False
    
    # Saving and logging
    save_total_limit: int = 10
    logging_steps: int = 10
    save_steps: int = 500
    fp16: bool = False
    bf16: bool = True
    max_grad_norm: float = 1.0
    
    def __post_init__(self):
        if self.lora_alpha is None:
            self.lora_alpha = self.lora_rank * 2
        
        # Set default tasks based on task_type if not provided
        if self.tasks is None:
            if self.task_type == "similar":
                self.tasks = ["sst2", "mnli", "qqp"]
            else:
                self.tasks = ["squad_v2", "codex_glue", "cnn_dailymail"]
        
        # Adjust batch size based on task type
        if self.task_type == "similar" and not self.use_lora:
            self.batch_size = 128  # Larger batch size for similar tasks with full fine-tuning

# Base models for each task type
SIMILAR_BASE_MODEL = "meta-llama/Llama-3.2-1B"  # Base LLaMA model for all training
DISSIMILAR_BASE_MODEL = "meta-llama/Llama-3.2-1B"  # Same base model, different task type

# Baseline configurations
SIMILAR_TASK_CONFIGS = [
    # Full fine-tuning baseline
    TrainingConfig(
        task_type="similar",
        tasks=["sst2", "mnli", "qqp"],
        base_model=SIMILAR_BASE_MODEL,
        use_lora=False,
        batch_size=128,
        epochs=3,
        learning_rate=1e-5,
        weight_decay=0,
        gradient_accumulation_steps=4,
    ),
    
    # LoRA configurations
    TrainingConfig(
        task_type="similar",
        tasks=["sst2", "mnli", "qqp"],
        base_model=SIMILAR_BASE_MODEL,
        use_lora=True,
        lora_rank=4,
        batch_size=32,
        epochs=3,
    ),
    TrainingConfig(
        task_type="similar",
        tasks=["sst2", "mnli", "qqp"],
        base_model=SIMILAR_BASE_MODEL,
        use_lora=True,
        lora_rank=8,
        batch_size=32,
        epochs=3,
    ),
    TrainingConfig(
        task_type="similar",
        tasks=["sst2", "mnli", "qqp"],
        base_model=SIMILAR_BASE_MODEL,
        use_lora=True,
        lora_rank=16,  # Original baseline
        batch_size=32,
        epochs=3,
    ),
    TrainingConfig(
        task_type="similar",
        tasks=["sst2", "mnli", "qqp"],
        base_model=SIMILAR_BASE_MODEL,
        use_lora=True,
        lora_rank=32,
        batch_size=32,
        epochs=3,
    ),
    TrainingConfig(
        task_type="similar",
        tasks=["sst2", "mnli", "qqp"],
        base_model=SIMILAR_BASE_MODEL,
        use_lora=True,
        lora_rank=64,
        batch_size=32,
        epochs=3,
    ),
]

DISSIMILAR_TASK_CONFIGS = [
    # Full fine-tuning baseline
    TrainingConfig(
        task_type="dissimilar",
        tasks=["squad_v2", "codex_glue", "cnn_dailymail"],
        base_model=DISSIMILAR_BASE_MODEL,
        use_lora=False,
        batch_size=16,
        epochs=3,
        learning_rate=1e-5,
        weight_decay=0,
        gradient_accumulation_steps=4,
    ),
    
    # LoRA configurations
    TrainingConfig(
        task_type="dissimilar",
        tasks=["squad_v2", "codex_glue", "cnn_dailymail"],
        base_model=DISSIMILAR_BASE_MODEL,
        use_lora=True,
        lora_rank=4,
        batch_size=2,
        gradient_accumulation_steps=16,  # Increased from 4 to 16 for effective batch size of 32
        epochs=3,
    ),
    TrainingConfig(
        task_type="dissimilar",
        tasks=["squad_v2", "codex_glue", "cnn_dailymail"],
        base_model=DISSIMILAR_BASE_MODEL,
        use_lora=True,
        lora_rank=8,
        batch_size=16,
        epochs=3,
    ),
    TrainingConfig(
        task_type="dissimilar",
        tasks=["squad_v2", "codex_glue", "cnn_dailymail"],
        base_model=DISSIMILAR_BASE_MODEL,
        use_lora=True,
        lora_rank=16,  # Original baseline
        batch_size=16,
        epochs=3,
    ),
    TrainingConfig(
        task_type="dissimilar",
        tasks=["squad_v2", "codex_glue", "cnn_dailymail"],
        base_model=DISSIMILAR_BASE_MODEL,
        use_lora=True,
        lora_rank=32,
        batch_size=16,
        epochs=3,
    ),
    TrainingConfig(
        task_type="dissimilar",
        tasks=["squad_v2", "codex_glue", "cnn_dailymail"],
        base_model=DISSIMILAR_BASE_MODEL,
        use_lora=True,
        lora_rank=64,
        batch_size=16,
        epochs=3,
    ),
]

def get_config_name(config: TrainingConfig) -> str:
    """Generate configuration name based on settings."""
    task_prefix = "similar" if config.task_type == "similar" else "dissimilar"
    if config.use_lora:
        return f"{task_prefix}_LoRA_{config.lora_rank}"
    return f"{task_prefix}_FullFT" 