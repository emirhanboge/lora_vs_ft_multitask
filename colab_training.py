"""
Training Script for Google Colab

This script sets up and runs the training pipeline in Google Colab.
It handles both similar and dissimilar task training with various LoRA configurations.
"""

import os
import torch
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Literal
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from datasets import load_dataset, concatenate_datasets, DatasetDict
from peft import get_peft_model, LoraConfig, TaskType
import evaluate

# Configuration Constants
OUTPUT_DIR = "outputs"
SIMILAR_TASKS = ["sst2", "mnli", "qqp"]
DISSIMILAR_TASKS = ["rajpurkar/squad_v2", "codex_glue", "cnn_dailymail"]
TASK_LABELS = {"sst2": 2, "mnli": 3, "qqp": 2}

DATASET_PATHS = {
    "sst2_mnli_qqp": "emirhanboge/sst2_mnli_qqp_llama1b_modified",
    "qa_code_summarization": "emirhanboge/qa_code_summarization_llama1b_modified",
}

TaskType = Literal["similar", "dissimilar"]

@dataclass
class TrainingConfig:
    """Configuration for training runs."""
    task_type: TaskType = "similar"
    model_name: str = "meta-llama/Llama-2-1b"
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: Optional[int] = None
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    batch_size: int = 32
    num_epochs: int = 3
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4
    save_total_limit: int = 2
    logging_steps: int = 10
    save_steps: int = 1000
    
    def __post_init__(self):
        if self.lora_alpha is None:
            self.lora_alpha = self.lora_rank * 2
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]
        if self.task_type == "similar" and not self.use_lora:
            self.batch_size = 128

def setup_environment():
    """Set up the Colab environment with necessary packages."""
    print("Setting up environment...")
    
    # Install required packages
    !pip install -q transformers==4.36.2 datasets==2.16.1 peft==0.7.1 accelerate==0.25.0 bitsandbytes==0.41.3 evaluate==0.4.1
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Environment setup complete!")

def get_config_name(config: TrainingConfig) -> str:
    """Generate configuration name based on settings."""
    task_prefix = "similar" if config.task_type == "similar" else "dissimilar"
    if config.use_lora:
        return f"{task_prefix}_LoRA_{config.lora_rank}"
    return f"{task_prefix}_FullFT"

class BaseTrainer:
    """Base class for all trainers."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_name = get_config_name(config)
    
    def get_training_args(self) -> TrainingArguments:
        """Get training arguments from config."""
        return TrainingArguments(
            output_dir=os.path.join(OUTPUT_DIR, self.model_name),
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=4,
            num_train_epochs=self.config.num_epochs,
            evaluation_strategy="epoch",
            save_strategy="steps",
            save_total_limit=self.config.save_total_limit,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            bf16=True,
            logging_dir=os.path.join(OUTPUT_DIR, "logs"),
            logging_steps=self.config.logging_steps,
            gradient_checkpointing=True,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            save_steps=self.config.save_steps,
            max_grad_norm=1.0,
            ddp_find_unused_parameters=False,
            optim="adamw_torch",
            fp16=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=2,  # Reduced for Colab
        )

    def get_lora_config(self, task_type: str) -> Optional[LoraConfig]:
        """Get LoRA configuration if enabled."""
        if not self.config.use_lora:
            return None
        return LoraConfig(
            task_type=task_type,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            target_modules=self.config.target_modules
        )

class SimilarTaskTrainer(BaseTrainer):
    """Handles training for similar classification tasks."""
    
    def get_model(self):
        print(f"Loading model from {self.config.model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=max(TASK_LABELS.values()),
            torch_dtype=torch.bfloat16,
        )

        if self.config.use_lora:
            print(f"Using LoRA (rank={self.config.lora_rank}, alpha={self.config.lora_alpha})")
            lora_config = self.get_lora_config(TaskType.SEQ_CLS)
            model = get_peft_model(model, lora_config)
            
        model.gradient_checkpointing_enable()
        model.config.pad_token_id = self.tokenizer.pad_token_id
        return model

    def load_dataset(self):
        print("Loading classification dataset...")
        return load_dataset(DATASET_PATHS["sst2_mnli_qqp"])

    @staticmethod
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = torch.tensor(predictions)
        if predictions.dim() == 3:
            predictions = predictions[:, 0, :]
        predicted_labels = torch.argmax(predictions, dim=-1)
        metric = evaluate.load("glue", "mnli")
        return metric.compute(predictions=predicted_labels, references=labels)

    def train(self):
        dataset = self.load_dataset()
        model = self.get_model()
        training_args = self.get_training_args()

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        print(f"Training {self.model_name}...")
        trainer.train()
        model.save_pretrained(os.path.join(OUTPUT_DIR, self.model_name))
        print(f"Finished training {self.model_name}")

class DissimilarTaskTrainer(BaseTrainer):
    """Handles training for dissimilar generation tasks."""
    
    def get_model(self):
        print(f"Loading model from {self.config.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_cache=False
        )

        if self.config.use_lora:
            print(f"Using LoRA (rank={self.config.lora_rank}, alpha={self.config.lora_alpha})")
            lora_config = self.get_lora_config(TaskType.CAUSAL_LM)
            model = get_peft_model(model, lora_config)
            
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.pad_token_id = self.tokenizer.pad_token_id
        return model

    def load_dataset(self):
        print("Loading generation dataset...")
        return load_dataset(DATASET_PATHS["qa_code_summarization"])

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_refs = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        results = {}
        
        # ROUGE metrics
        rouge_metric = evaluate.load("rouge")
        rouge_result = rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_refs
        )
        results.update({
            "ROUGE-1": round(rouge_result["rouge1"].mid.fmeasure * 100, 2),
            "ROUGE-2": round(rouge_result["rouge2"].mid.fmeasure * 100, 2),
            "ROUGE-L": round(rouge_result["rougeL"].mid.fmeasure * 100, 2)
        })

        return results

    def train(self):
        dataset = self.load_dataset()
        torch.cuda.empty_cache()
        model = self.get_model()
        training_args = self.get_training_args()

        class CustomDataCollator:
            def __init__(self, tokenizer, max_length=1024):
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __call__(self, features):
                input_ids = [f["input_ids"] for f in features]
                labels = [f["labels"] for f in features]

                input_ids = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(x[:self.max_length]) for x in input_ids],
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id
                )
                labels = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(x[:self.max_length]) for x in labels],
                    batch_first=True,
                    padding_value=-100
                )

                attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                }

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=CustomDataCollator(self.tokenizer),
            compute_metrics=self.compute_metrics,
        )

        print(f"Training {self.model_name}...")
        trainer.train()
        model.save_pretrained(os.path.join(OUTPUT_DIR, self.model_name))
        print(f"Finished training {self.model_name}")

def main():
    """Main training function."""
    # Set up the environment
    setup_environment()
    
    # Example configurations
    similar_configs = [
        TrainingConfig(task_type="similar", use_lora=True, lora_rank=rank)
        for rank in [4, 8, 16, 32, 64]
    ]
    
    dissimilar_configs = [
        TrainingConfig(task_type="dissimilar", use_lora=True, lora_rank=rank)
        for rank in [4, 8, 16, 32, 64]
    ]
    
    # Choose configuration
    from IPython.display import display
    import ipywidgets as widgets
    
    task_dropdown = widgets.Dropdown(
        options=['similar', 'dissimilar'],
        description='Task Type:',
    )
    
    rank_dropdown = widgets.Dropdown(
        options=[4, 8, 16, 32, 64],
        description='LoRA Rank:',
    )
    
    display(task_dropdown)
    display(rank_dropdown)
    
    def on_button_clicked(b):
        config = TrainingConfig(
            task_type=task_dropdown.value,
            use_lora=True,
            lora_rank=rank_dropdown.value
        )
        
        if config.task_type == "similar":
            trainer = SimilarTaskTrainer(config)
        else:
            trainer = DissimilarTaskTrainer(config)
            
        trainer.train()
    
    button = widgets.Button(description="Start Training")
    button.on_click(on_button_clicked)
    display(button)

if __name__ == "__main__":
    main() 