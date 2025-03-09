"""
Multi-task training script 

This script handles fine-tuning LLaMA models on multiple tasks simultaneously,
supporting both similar tasks (classification) and dissimilar tasks (generation).
"""

import os
import torch
import json
from dataclasses import asdict
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset, concatenate_datasets, DatasetDict, Value
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
from config import (
    DATA_DIR,
    OUTPUT_DIR,
    TOKENIZER_PATH,
    TASK_LABELS,
    METRIC_MAPPING,
    SIMILAR_TASKS,
    DISSIMILAR_TASKS,
    DATASET_PATHS,
    MAX_LENGTHS,
)
from training_configs import (
    TrainingConfig,
    SIMILAR_TASK_CONFIGS,
    DISSIMILAR_TASK_CONFIGS,
    get_config_name,
)

from typing import Optional

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

#######################
# Common Utilities
#######################

def setup_tokenizer(model_name: str):
    """Initialize and configure the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def model_already_trained(model_name):
    """Check if the fine-tuned model already exists."""
    return os.path.exists(os.path.join(OUTPUT_DIR, model_name))

#######################
# Base Trainer
#######################

class BaseTrainer:
    """Base class for all trainers."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = setup_tokenizer(config.model_name)
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
            dataloader_num_workers=4,
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

#######################
# Similar Tasks (Classification)
#######################

class SimilarTaskTrainer(BaseTrainer):
    """Handles training for similar classification tasks (SST2, MNLI, QQP)."""
    
    def get_model(self):
        """Load classification model."""
        print(f"Loading model from {self.config.model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=max(TASK_LABELS.values())
        )

        if self.config.use_lora:
            print(f"Using LoRA (rank={self.config.lora_rank}, alpha={self.config.lora_alpha})")
            lora_config = self.get_lora_config(TaskType.SEQ_CLS)
            model = get_peft_model(model, lora_config)

        model.config.pad_token_id = self.tokenizer.pad_token_id
        model.resize_token_embeddings(len(self.tokenizer))
        return model

    def load_dataset(self):
        """Load multi-task classification dataset from Hugging Face Hub."""
        print("Loading multi-task classification dataset...")
        
        # Load the combined dataset directly
        try:
            dataset = load_dataset(DATASET_PATHS["sst2_mnli_qqp"])
            print("Successfully loaded combined dataset from Hugging Face Hub")
            return dataset
        except Exception as e:
            print(f"Error loading combined dataset: {e}")
            print("Falling back to loading individual datasets...")
        
        # Fallback: Load and combine individual datasets
        datasets = []
        for task in SIMILAR_TASKS:
            try:
                dataset = load_dataset(DATASET_PATHS[task])
                print(f"Loaded {task} dataset")
                datasets.append(dataset)
            except Exception as e:
                print(f"Error loading {task} dataset: {e}")
                continue

        if not datasets:
            raise ValueError("No datasets could be loaded!")

        # Merge datasets
        train_datasets = [ds["train"] for ds in datasets]
        validation_datasets = []
        
        for ds in datasets:
            if "validation" in ds:
                validation_datasets.append(ds["validation"])
            if "validation_matched" in ds:
                validation_datasets.append(ds["validation_matched"])
            if "validation_mismatched" in ds:
                validation_datasets.append(ds["validation_mismatched"])

        return DatasetDict({
            "train": concatenate_datasets(train_datasets),
            "validation": concatenate_datasets(validation_datasets),
        })

    @staticmethod
    def compute_metrics(eval_pred):
        """Compute classification metrics."""
        predictions, labels = eval_pred
        if labels is None or len(labels) == 0:
            raise ValueError("Labels are missing in the dataset.")

        predictions = torch.tensor(predictions)
        if predictions.dim() == 3:
            predictions = predictions[:, 0, :]

        predicted_labels = torch.argmax(predictions, dim=-1)
        metric = evaluate.load("glue", "mnli")  # Using MNLI metric for all
        return metric.compute(predictions=predicted_labels, references=labels)

    def train(self):
        """Train on similar classification tasks."""
        if model_already_trained(self.model_name):
            print(f"Model {self.model_name} already exists, skipping training.")
            return

        dataset = self.load_dataset()
        if dataset is None:
            return

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

#######################
# Dissimilar Tasks (Generation)
#######################

class DissimilarTaskTrainer(BaseTrainer):
    """Handles training for dissimilar generation tasks (QA, Code, Summarization)."""
    
    def get_model(self):
        """Load generation model."""
        print(f"Loading model from {self.config.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            device_map="auto",
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
        model.resize_token_embeddings(len(self.tokenizer))
        return model

    def load_dataset(self):
        """Load multi-task generation dataset from Hugging Face Hub."""
        print("Loading multi-task generation dataset...")
        
        # Load the combined dataset directly
        try:
            dataset = load_dataset(DATASET_PATHS["qa_code_summarization"])
            print("Successfully loaded combined dataset from Hugging Face Hub")
            return dataset
        except Exception as e:
            print(f"Error loading combined dataset: {e}")
            print("Falling back to loading individual datasets...")
            
            # Fallback: Load and combine individual datasets
            datasets = []
            for task in DISSIMILAR_TASKS:
                try:
                    dataset = load_dataset(DATASET_PATHS[task])
                    print(f"Loaded {task} dataset")
                    datasets.append(dataset)
                except Exception as e:
                    print(f"Error loading {task} dataset: {e}")
                    continue

            if not datasets:
                raise ValueError("No datasets could be loaded!")

            # Merge datasets
            train_datasets = [ds["train"] for ds in datasets]
            validation_datasets = [
                ds["validation"] for ds in datasets 
                if "validation" in ds and ds["validation"] is not None
            ]

            return DatasetDict({
                "train": concatenate_datasets(train_datasets),
                "validation": concatenate_datasets(validation_datasets) if validation_datasets else None,
            })

    def compute_metrics(self, eval_pred):
        """Compute generation metrics."""
        predictions, labels = eval_pred
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_refs = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        results = {}

        # QA metrics (squad_v2)
        squad_metric = evaluate.load("squad_v2")
        squad_result = squad_metric.compute(
            predictions=[
                {"id": str(i), "prediction_text": p}
                for i, p in enumerate(decoded_preds)
            ],
            references=[
                {"id": str(i), "answers": {"answer_start": [0], "text": [r]}}
                for i, r in enumerate(decoded_refs)
            ]
        )
        results.update({
            "squad_v2_EM": squad_result["exact"],
            "squad_v2_F1": squad_result["f1"]
        })

        # ROUGE metrics for all tasks
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

        # BLEU metric for all tasks
        bleu_metric = evaluate.load("bleu")
        bleu_result = bleu_metric.compute(
            predictions=[p.split() for p in decoded_preds],
            references=[[r.split()] for r in decoded_refs]
        )
        results["BLEU"] = round(bleu_result["bleu"] * 100, 2)

        return results

    def train(self):
        """Train on dissimilar generation tasks."""
        if model_already_trained(self.model_name):
            print(f"Model {self.model_name} already exists, skipping training.")
            return

        dataset = self.load_dataset()
        if dataset is None:
            return

        torch.cuda.empty_cache()
        
        model = self.get_model()
        training_args = self.get_training_args()

        class CustomDataCollator:
            def __init__(self, tokenizer, max_length=2048):
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

        data_collator = CustomDataCollator(self.tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        print(f"Training {self.model_name}...")
        trainer.train()
        
        model.save_pretrained(os.path.join(OUTPUT_DIR, self.model_name))
        print(f"Finished training {self.model_name}")

def run_training_configuration(config: TrainingConfig):
    """Run training with a specific configuration."""
    # Save configuration for reproducibility
    os.makedirs(os.path.join(OUTPUT_DIR, "configs"), exist_ok=True)
    config_path = os.path.join(OUTPUT_DIR, "configs", f"{get_config_name(config)}.json")
    
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    torch.cuda.empty_cache()
    
    if config.task_type == "similar":
        trainer = SimilarTaskTrainer(config)
        trainer.train()
    else:
        trainer = DissimilarTaskTrainer(config)
        trainer.train()

def main():
    """Main function to run training configurations."""
    import argparse
    parser = argparse.ArgumentParser(description="Run multi-task training with specific configuration")
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["similar", "dissimilar"],
        required=True,
        help="Type of tasks to train on"
    )
    parser.add_argument(
        "--config-index",
        type=int,
        required=True,
        help="Index of the configuration to use (0-5, where 0 is full fine-tuning and 1-5 are LoRA ranks 4,8,16,32,64)"
    )
    args = parser.parse_args()
    
    configs = SIMILAR_TASK_CONFIGS if args.task_type == "similar" else DISSIMILAR_TASK_CONFIGS
    if args.config_index < 0 or args.config_index >= len(configs):
        raise ValueError(f"Config index must be between 0 and {len(configs)-1}")
    
    config = configs[args.config_index]
    run_training_configuration(config)

if __name__ == "__main__":
    main()