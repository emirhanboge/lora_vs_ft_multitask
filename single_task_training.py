"""
Single-task training script

This script handles fine-tuning LLaMA models on individual classification tasks
using either full fine-tuning or LoRA.
"""

import os
import torch
import json
from dataclasses import asdict
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from datasets import load_from_disk
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
from config import (
    DATA_DIR,
    MODEL_NAME,
    OUTPUT_DIR,
    TOKENIZER_PATH,
    SIMILAR_TASKS,
    TASK_LABELS,
)
from training_configs import (
    TrainingConfig,
    SIMILAR_TASK_CONFIGS,
    get_config_name,
)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def setup_tokenizer():
    """Initialize and configure the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

class SingleTaskTrainer:
    """Handles training for individual classification tasks."""
    
    def __init__(self, task: str, config: TrainingConfig):
        self.task = task
        self.config = config
        self.tokenizer = setup_tokenizer()
        self.model_name = f"single_{task}_{get_config_name(config)}"
        
    def get_model(self):
        """Load classification model."""
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=TASK_LABELS[self.task]
        )

        if self.config.use_lora:
            print(f"Using LoRA (rank={self.config.lora_rank}, alpha={self.config.lora_alpha})")
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                target_modules=self.config.target_modules
            )
            model = get_peft_model(model, peft_config)

        model.config.pad_token_id = self.tokenizer.pad_token_id
        model.resize_token_embeddings(len(self.tokenizer))
        return model

    def get_training_args(self) -> TrainingArguments:
        """Get training arguments from config."""
        return TrainingArguments(
            output_dir=os.path.join(OUTPUT_DIR, self.model_name),
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=4,
            num_train_epochs=self.config.num_epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            bf16=True,
            logging_dir=os.path.join(OUTPUT_DIR, "logs"),
            logging_steps=self.config.logging_steps,
            save_total_limit=self.config.save_total_limit,
            gradient_checkpointing=False,
            optim="adamw_torch",
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )

    def load_dataset(self):
        """Load dataset for the task."""
        dataset_path = os.path.join(DATA_DIR, self.task.replace("/", "_"))
        if not os.path.exists(dataset_path):
            print(f"Dataset {self.task} not found at {dataset_path}")
            return None

        dataset = load_from_disk(dataset_path)

        # Fix labels if needed
        if "labels" in dataset["train"].features and isinstance(dataset["train"][0]["labels"], list):
            print("Fixing labels...")
            dataset = dataset.map(lambda x: {"labels": x["labels"][0]})

        return dataset

    def compute_metrics(self, eval_pred):
        """Compute metrics for the task."""
        predictions, labels = eval_pred
        if labels is None or len(labels) == 0:
            raise ValueError("Labels are missing in the dataset.")

        predictions = torch.tensor(predictions)
        if predictions.dim() == 3:
            predictions = predictions[:, 0, :]

        predicted_labels = torch.argmax(predictions, dim=-1)
        metric = evaluate.load("glue", self.task) if self.task in ["sst2", "mnli", "qqp"] else evaluate.load(self.task)
        return metric.compute(predictions=predicted_labels, references=labels)

    def train(self):
        """Train on the task."""
        if os.path.exists(os.path.join(OUTPUT_DIR, self.model_name)):
            print(f"Model {self.model_name} already exists, skipping training.")
            return

        dataset = self.load_dataset()
        if dataset is None:
            return

        model = self.get_model()
        training_args = self.get_training_args()

        eval_dataset = dataset.get("validation", dataset.get("validation_matched", None))

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        print(f"🚀 Training {self.task}...")
        trainer.train()

        model.save_pretrained(os.path.join(OUTPUT_DIR, self.model_name))
        print(f"✅ Finished training {self.task}\n")

def run_training_configuration(task: str, config: TrainingConfig):
    """Run training with a specific configuration."""
    # Save configuration for reproducibility
    os.makedirs(os.path.join(OUTPUT_DIR, "configs"), exist_ok=True)
    config_path = os.path.join(OUTPUT_DIR, "configs", f"single_{task}_{get_config_name(config)}.json")
    
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    torch.cuda.empty_cache()
    
    trainer = SingleTaskTrainer(task, config)
    trainer.train()

def main():
    """Main function to run training configurations."""
    import argparse
    parser = argparse.ArgumentParser(description="Run single-task training with specific configuration")
    parser.add_argument(
        "--task",
        type=str,
        choices=SIMILAR_TASKS,
        required=True,
        help="Task to train on"
    )
    parser.add_argument(
        "--config-index",
        type=int,
        required=True,
        help="Index of the configuration to use (0-5, where 0 is full fine-tuning and 1-5 are LoRA ranks 4,8,16,32,64)"
    )
    args = parser.parse_args()
    
    if args.config_index < 0 or args.config_index >= len(SIMILAR_TASK_CONFIGS):
        raise ValueError(f"Config index must be between 0 and {len(SIMILAR_TASK_CONFIGS)-1}")
    
    config = SIMILAR_TASK_CONFIGS[args.config_index]
    run_training_configuration(args.task, config)

if __name__ == "__main__":
    main()
