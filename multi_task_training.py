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
from datasets import load_dataset, concatenate_datasets, DatasetDict, Value, Dataset, Sequence
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
    EXAMPLES_PER_TASK,
)
from training_configs import (
    TrainingConfig,
    SIMILAR_TASK_CONFIGS,
    DISSIMILAR_TASK_CONFIGS,
    get_config_name,
)
from huggingface_hub import HfApi
from datasets import load_from_disk
import logging
from typing import Optional, Dict, List, Union, Any, Tuple
from dataset_descriptions import DATASET_DESCRIPTIONS
import argparse
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Hugging Face Hub credentials
HF_USERNAME = os.environ.get("HF_USERNAME", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

def setup_tokenizer(model_name: str):
    """Set up the tokenizer for the given model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer

def model_already_trained(model_name):
    """Check if a model has already been trained."""
    return os.path.exists(os.path.join(OUTPUT_DIR, model_name))

def check_dataset_on_hub(dataset_name: str) -> bool:
    """Check if a dataset exists on the Hugging Face Hub."""
    try:
        api = HfApi()
        dataset_info = api.dataset_info(f"{HF_USERNAME}/{dataset_name}")
        return True
    except Exception:
        return False

class BaseTrainer:
    """Base class for training models on different tasks."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize the trainer with the given configuration."""
        self.config = config
        self.tokenizer = setup_tokenizer(config.base_model)
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        
    def get_training_args(self) -> TrainingArguments:
        """Get the training arguments for the Trainer."""
        output_dir = os.path.join(OUTPUT_DIR, get_config_name(self.config))
        
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            max_grad_norm=self.config.max_grad_norm,
            warmup_ratio=self.config.warmup_ratio,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            push_to_hub=False,
            report_to="none",  # Disable all logging backends
            label_names=["labels"],  # Add label_names parameter
        )
    
    def get_lora_config(self, task_type: str) -> Optional[LoraConfig]:
        """Get the LoRA configuration if LoRA is enabled."""
        if not self.config.use_lora:
            return None
        
        return LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=task_type,
            target_modules=["q_proj", "v_proj"],
        )

class SimilarTaskTrainer(BaseTrainer):
    """Trainer for similar tasks (classification)."""
    
    def get_model(self):
        """Get the model for similar tasks."""
        num_labels = len(TASK_LABELS[self.config.tasks[0]])
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.base_model,
            num_labels=num_labels,
        )
        
        lora_config = self.get_lora_config(TaskType.SEQ_CLS)
        if lora_config:
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        return model
    
    def load_dataset(self):
        """Load and preprocess datasets for similar tasks."""
        datasets = []
        
        for task in self.config.tasks:
            dataset = load_dataset(DATASET_PATHS[task])
            
            # Preprocess dataset
            def preprocess_function(examples):
                texts = examples["sentence"] if "sentence" in examples else examples["sentence1"]
                if "sentence2" in examples:
                    texts = [f"{t1} [SEP] {t2}" for t1, t2 in zip(texts, examples["sentence2"])]
                
                tokenized = self.tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_LENGTHS[task],
                    return_tensors="pt",
                )
                
                tokenized["labels"] = examples["label"]
                return tokenized
            
            processed_dataset = dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=dataset["train"].column_names,
            )
            
            datasets.append(processed_dataset)
        
        # Combine datasets
        combined_train = concatenate_datasets([ds["train"] for ds in datasets])
        combined_eval = concatenate_datasets([ds["validation"] for ds in datasets])
        
        self.train_dataset = combined_train
        self.eval_dataset = combined_eval
    
    @staticmethod
    def compute_metrics(eval_pred):
        """Compute metrics for similar tasks."""
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=1)
        
        accuracy = evaluate.load("accuracy")
        return accuracy.compute(predictions=predictions, references=labels)
    
    def train(self):
        """Train the model on similar tasks."""
        if model_already_trained(get_config_name(self.config)):
            logger.info(f"Model {get_config_name(self.config)} already trained, skipping...")
            return
        
        logger.info(f"Training model on tasks: {self.config.tasks}")
        
        self.model = self.get_model()
        self.load_dataset()
        
        training_args = self.get_training_args()
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        # Save config
        with open(os.path.join(training_args.output_dir, "config.json"), "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        
        logger.info(f"Training complete. Model saved to {training_args.output_dir}")

class DissimilarTaskTrainer(BaseTrainer):
    """Trainer for dissimilar tasks (generation)."""
    
    def get_model(self):
        """Get the model for dissimilar tasks."""
        # Clear CUDA cache before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            use_cache=False,  # Disable KV cache for gradient checkpointing
        )
        
        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
        
        lora_config = self.get_lora_config(TaskType.CAUSAL_LM)
        if lora_config:
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        return model
    
    def load_dataset(self):
        """Load and preprocess datasets for dissimilar tasks."""
        # Get the combined dataset name
        combined_dataset_name = self._get_combined_dataset_name()
        
        # Load combined dataset from Hub
        logger.info(f"Loading combined dataset from Hugging Face Hub: {combined_dataset_name}")
        try:
            combined_dataset = load_dataset(f"{HF_USERNAME}/{combined_dataset_name}")
            
            # Sample EXAMPLES_PER_TASK examples for each task
            sampled_train_datasets = []
            sampled_eval_datasets = []
            
            for task in self.config.tasks:
                # Filter train dataset for current task
                task_train = combined_dataset["train"].filter(lambda x: x["task"] == task)
                if len(task_train) > EXAMPLES_PER_TASK:
                    # Randomly sample EXAMPLES_PER_TASK examples
                    task_train = task_train.shuffle(seed=42).select(range(EXAMPLES_PER_TASK))
                sampled_train_datasets.append(task_train)
                
                # Filter validation dataset for current task
                task_val = combined_dataset["validation"].filter(lambda x: x["task"] == task)
                # For validation, take 10% of EXAMPLES_PER_TASK or all examples if less
                val_size = min(len(task_val), EXAMPLES_PER_TASK // 10)
                if len(task_val) > val_size:
                    task_val = task_val.shuffle(seed=42).select(range(val_size))
                sampled_eval_datasets.append(task_val)
            
            # Combine sampled datasets
            self.train_dataset = concatenate_datasets(sampled_train_datasets)
            self.eval_dataset = concatenate_datasets(sampled_eval_datasets)
            
            logger.info(f"Successfully loaded and sampled dataset from Hub: {combined_dataset_name}")
            logger.info(f"Train dataset size: {len(self.train_dataset)}, Validation dataset size: {len(self.eval_dataset)}")
            return
            
        except Exception as e:
            logger.error(f"Error loading dataset from Hub: {str(e)}")
            logger.error("Please run with --prepare-datasets-only first to create and upload the datasets to the Hub")
            raise
    
    def _get_combined_dataset_name(self) -> str:
        """Get the name for the combined dataset based on tasks."""
        if set(self.config.tasks) == {"squad_v2", "codex_glue"}:
            return "squad_v2_codex_glue_llama1b_modified"
        elif set(self.config.tasks) == {"squad_v2", "codex_glue", "cnn_dailymail"}:
            return "squad_v2_codex_glue_cnn_dailymail_llama1b_modified"
        else:
            # Create a name based on the tasks
            task_names = [t.split("/")[-1] for t in self.config.tasks]
            return f"{'_'.join(task_names)}_llama1b_modified"
    
    def _process_dataset(self, dataset: Dataset, task: str, force_process: bool = False) -> Dataset:
        """Process a dataset for a specific task."""
        # Check if dataset is already processed, unless force_process is True
        if not force_process and all(col in dataset.column_names for col in ['input_ids', 'attention_mask', 'labels', 'task', 'raw_inputs', 'raw_labels']):
            logger.info(f"Dataset already processed for task {task}, skipping processing...")
            return dataset.remove_columns([col for col in dataset.column_names 
                                        if col not in ['input_ids', 'attention_mask', 'labels', 'task', 'raw_inputs', 'raw_labels']])
            
        # Define task prefix
        task_prefix = f"Task: {task} | "
        
        # Process examples based on task type
        def process_examples(examples):
            input_texts = []
            labels = []
            
            if task == "squad_v2":  # Question Answering
                input_texts = [
                    f"{task_prefix} Read the passage and answer. Context: {c} Question: {q}"
                    for c, q in zip(examples["context"], examples["question"])
                ]
                labels = [a["text"][0] if a["text"] else "" for a in examples["answers"]]
            
            elif task == "codex_glue":  # Code Generation
                input_texts = [
                    f"{task_prefix} Generate a description for the following code snippet:\n{code}"
                    for code in examples["original_string"]
                ]
                labels = examples["docstring"]
            
            elif task == "cnn_dailymail":  # Summarization
                input_texts = [
                    f"{task_prefix} Summarize this news article:\n{article}"
                    for article in examples["article"]
                ]
                labels = examples["highlights"]
            
            return {
                "input_texts": input_texts,
                "labels": labels,
                "raw_inputs": input_texts,
                "raw_labels": labels
            }
        
        logger.info(f"Processing examples for task {task}...")
        # Process the examples with progress bar
        processed = dataset.map(
            process_examples,
            batched=True,
            remove_columns=dataset.column_names,
            desc=f"Processing {task} examples",
            load_from_cache_file=False  # Don't use cache to ensure fresh processing
        )
        
        # Tokenize the processed examples
        def tokenize_function(examples):
            # Tokenize inputs without padding
            model_inputs = self.tokenizer(
                examples["input_texts"],
                truncation=True,
                padding=False,
                add_special_tokens=True,  # This adds BOS token
                return_attention_mask=True,
                verbose=False  # Disable tokenizer warnings
            )
            
            # Manually add EOS token to input_ids if not present
            for i in range(len(model_inputs["input_ids"])):
                if model_inputs["input_ids"][i][-1] != self.tokenizer.eos_token_id:
                    model_inputs["input_ids"][i].append(self.tokenizer.eos_token_id)
                    model_inputs["attention_mask"][i].append(1)
            
            # Tokenize labels without padding
            labels = self.tokenizer(
                examples["labels"],
                truncation=True,
                padding=False,
                add_special_tokens=False,  # We'll add EOS manually
                verbose=False  # Disable tokenizer warnings
            )
            
            # Manually add EOS token to labels
            model_inputs["labels"] = [
                label_ids + [self.tokenizer.eos_token_id]
                for label_ids in labels["input_ids"]
            ]
            
            # Add task field and raw texts
            model_inputs["task"] = [task] * len(examples["input_texts"])
            model_inputs["raw_inputs"] = examples["raw_inputs"]
            model_inputs["raw_labels"] = examples["raw_labels"]
            
            return model_inputs
        
        logger.info(f"Tokenizing examples for task {task}...")
        # Apply tokenization with progress bar
        tokenized_dataset = processed.map(
            tokenize_function,
            batched=True,
            remove_columns=["input_texts", "labels"],
            desc=f"Tokenizing {task} examples",
            load_from_cache_file=False  # Don't use cache to ensure fresh processing
        )
        
        # Cast features to ensure consistency
        logger.info(f"Casting features for task {task}...")
        feature_dtypes = {
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int8")),
            "labels": Sequence(Value("int64")),
            "task": Value("string"),
            "raw_inputs": Value("string"),
            "raw_labels": Value("string")
        }
        
        for col, dtype in feature_dtypes.items():
            tokenized_dataset = tokenized_dataset.cast_column(col, dtype)
        
        logger.info(f"Completed processing dataset for task {task}")
        return tokenized_dataset
    
    def _upload_dataset(self, dataset_name: str, dataset_path: str):
        """Upload a dataset to the Hugging Face Hub."""
        if not HF_USERNAME or not HF_TOKEN:
            logger.warning("HF_USERNAME or HF_TOKEN not set, skipping dataset upload")
            return
        
        logger.info(f"Uploading dataset {dataset_name} to Hugging Face Hub...")
        
        try:
            # Load dataset from disk
            dataset = load_from_disk(dataset_path)
            
            # Create repository on Hugging Face with correct repo type
            api = HfApi()
            repo_id = f"{HF_USERNAME}/{dataset_name}"
            api.create_repo(repo_id, exist_ok=True, token=HF_TOKEN, repo_type="dataset")
            
            # Push dataset to Hub
            dataset.push_to_hub(repo_id, token=HF_TOKEN)
            
            # Create and upload README
            readme_content = DATASET_DESCRIPTIONS.get(
                dataset_name.replace("_llama1b_modified", ""),
                f"# {dataset_name}\n\nPreprocessed dataset for LLaMA 1B model training."
            )
            
            readme_path = os.path.join(dataset_path, "README.md")
            with open(readme_path, "w") as f:
                f.write(readme_content)
            
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                token=HF_TOKEN,
                repo_type="dataset"
            )
            
            logger.info(f"Successfully uploaded {dataset_name} to {repo_id}")
        
        except Exception as e:
            logger.error(f"Error uploading dataset {dataset_name}: {str(e)}")
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for dissimilar tasks."""
        predictions, labels = eval_pred
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 with pad token id
        labels = torch.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute ROUGE scores
        rouge = evaluate.load("rouge")
        rouge_scores = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        
        # Compute BLEU score
        bleu = evaluate.load("bleu")
        bleu_score = bleu.compute(
            predictions=decoded_preds,
            references=[[label] for label in decoded_labels],
        )
        
        # Combine metrics
        metrics = {
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"],
            "bleu": bleu_score["bleu"],
        }
        
        return metrics
    
    def train(self):
        """Train the model on dissimilar tasks."""
        if model_already_trained(get_config_name(self.config)):
            logger.info(f"Model {get_config_name(self.config)} already trained, skipping...")
            return
        
        logger.info(f"Training model on tasks: {self.config.tasks}")
        
        self.model = self.get_model()
        self.load_dataset()
        
        training_args = self.get_training_args()
        
        # Custom collate function
        def custom_data_collator(features):
            # Convert list of dicts to dict of lists
            batch = {
                k: [feature[k] for feature in features]
                for k in features[0].keys()
            }
            
            # Determine max length in this batch
            max_length = max(len(ids) for ids in batch["input_ids"])
            
            # Initialize tensors
            input_ids = []
            attention_mask = []
            labels = []
            
            # Pad sequences
            for i in range(len(features)):
                # Pad input_ids
                padding_length = max_length - len(batch["input_ids"][i])
                input_ids.append(
                    batch["input_ids"][i] + [self.tokenizer.pad_token_id] * padding_length
                )
                attention_mask.append(
                    batch["attention_mask"][i] + [0] * padding_length
                )
                
                # Pad labels
                label_padding_length = max_length - len(batch["labels"][i])
                labels.append(
                    batch["labels"][i] + [-100] * label_padding_length
                )
            
            # Convert to tensors
            batch = {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels)
            }
            
            return batch
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=custom_data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        # Save config
        with open(os.path.join(training_args.output_dir, "config.json"), "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        
        logger.info(f"Training complete. Model saved to {training_args.output_dir}")

def run_training_configuration(config: TrainingConfig):
    """Run a specific training configuration."""
    if all(task in SIMILAR_TASKS for task in config.tasks):
        trainer = SimilarTaskTrainer(config)
    elif all(task in DISSIMILAR_TASKS for task in config.tasks):
        trainer = DissimilarTaskTrainer(config)
    else:
        raise ValueError(f"Mixed task types not supported: {config.tasks}")
    
    trainer.train()

def upload_model_to_hub(model_name, tasks, lora_rank=None):
    """Upload fine-tuned models to Hugging Face Hub."""
    if not HF_USERNAME or not HF_TOKEN:
        logger.warning("HF_USERNAME or HF_TOKEN not set, skipping model upload")
        return
    
    model_path = os.path.join(OUTPUT_DIR, model_name)
    repo_id = f"{HF_USERNAME}/{model_name}"
    
    logger.info(f"Uploading {model_name} to {repo_id}...")
    
    try:
        # Create repository if it doesn't exist
        api = HfApi()
        api.create_repo(repo_id, exist_ok=True, token=HF_TOKEN)
        
        # Determine model type based on tasks
        if all(task in SIMILAR_TASKS for task in tasks):
            # Classification model
            num_labels = len(TASK_LABELS[tasks[0]])
            model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        else:
            # Generation model
            model = AutoModelForCausalLM.from_pretrained(model_path)
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Save model to HF Hub
        model.push_to_hub(repo_id, token=HF_TOKEN)
        tokenizer.push_to_hub(repo_id, token=HF_TOKEN)
        
        # Add model card
        task_names = [t.split("/")[-1] for t in tasks]
        model_card = f"""# {model_name}

Fine-tuned LLaMA model on {', '.join(task_names)} {'dataset' if len(tasks) == 1 else 'datasets'}.

- **LoRA**: {'Enabled' if 'LoRA' in model_name else 'Full Fine-Tuning'}
- **LoRA Rank**: {lora_rank if lora_rank else 'N/A'}
- **Tasks**: {', '.join(task_names)}
- **Base Model**: LLaMA 1B
- **Model Type**: {'Classification' if all(task in SIMILAR_TASKS for task in tasks) else 'Generation'}

Trained using the 🤗 Transformers `Trainer` API.
"""
        
        with open(os.path.join(model_path, "README.md"), "w") as f:
            f.write(model_card)
        
        api.upload_file(
            path_or_fileobj=os.path.join(model_path, "README.md"),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=HF_TOKEN
        )
        
        logger.info(f"Successfully uploaded {model_name} to {repo_id}")
    
    except Exception as e:
        logger.error(f"Error uploading model {model_name}: {str(e)}")

def prepare_and_upload_datasets(tasks, force_upload=False):
    """Prepare and upload datasets for the specified tasks to the Hugging Face Hub."""
    logger.info(f"Preparing datasets for tasks: {tasks}")
    
    # Create a temporary trainer to handle dataset processing
    config = TrainingConfig(tasks=tasks)
    trainer = DissimilarTaskTrainer(config)
    
    # Process and combine datasets
    all_train_datasets = []
    all_eval_datasets = []
    
    # First, process and upload individual datasets
    for task in tasks:
        logger.info(f"Processing dataset for task: {task}")
        
        try:
            # Load the dataset
            if task == "codex_glue":
                dataset = load_dataset("google/code_x_glue_ct_code_to_text", "java")
            elif task == "cnn_dailymail":
                dataset = load_dataset("cnn_dailymail", "3.0.0")
            else:
                dataset = load_dataset(DATASET_PATHS[task])
            
            # Process train split with force_process=True to ensure processing
            train_processed = trainer._process_dataset(
                dataset["train"], 
                task, 
                force_process=True  # Force processing
            )
            
            # Process validation split with force_process=True
            val_processed = trainer._process_dataset(
                dataset["validation"], 
                task, 
                force_process=True  # Force processing
            )
            
            # Create a DatasetDict for the individual task
            task_dataset = DatasetDict({
                "train": train_processed,
                "validation": val_processed
            })
            
            # Save the individual dataset locally
            task_dataset_path = os.path.join(DATA_DIR, f"{task}_llama1b_modified")
            task_dataset.save_to_disk(task_dataset_path)
            
            # Upload individual dataset
            trainer._upload_dataset(f"{task}_llama1b_modified", task_dataset_path)
            
            # Add to combined datasets
            all_train_datasets.append(train_processed)
            all_eval_datasets.append(val_processed)
            
            logger.info(f"Successfully processed and uploaded dataset for task: {task}")
            
        except Exception as e:
            logger.error(f"Error processing dataset for task {task}: {str(e)}")
            raise
    
    # Create multitask dataset if we have more than one task
    if len(tasks) > 1:
        logger.info("Creating and uploading multitask dataset...")
        
        # Combine datasets
        combined_train = concatenate_datasets(all_train_datasets)
        combined_eval = concatenate_datasets(all_eval_datasets)
        
        # Create DatasetDict
        combined_dataset = DatasetDict({
            "train": combined_train,
            "validation": combined_eval
        })
        
        # Get the combined dataset name
        combined_dataset_name = trainer._get_combined_dataset_name()
        
        # Save locally
        combined_dataset_path = os.path.join(DATA_DIR, combined_dataset_name.split("/")[-1])
        combined_dataset.save_to_disk(combined_dataset_path)
        
        # Upload combined dataset
        trainer._upload_dataset(combined_dataset_name, combined_dataset_path)
        
        logger.info(f"Successfully uploaded combined dataset: {combined_dataset_name}")
        return combined_dataset_name
    
    return None

def main():
    """Main function to run training configurations."""
    parser = argparse.ArgumentParser(description="Train models on similar or dissimilar tasks")
    parser.add_argument("--task-type", type=str, choices=["similar", "dissimilar", "all"], default="dissimilar",
                        help="Type of tasks to train on (default: dissimilar)")
    parser.add_argument("--lora-rank", type=int, default=None,
                        help="Specific LoRA rank to train (e.g., 4, 8, 16, 32, 64). If not specified, all ranks will be trained.")
    parser.add_argument("--full-ft", action="store_true", 
                        help="Train with full fine-tuning instead of LoRA")
    parser.add_argument("--upload", action="store_true",
                        help="Upload models and datasets to Hugging Face Hub after training")
    parser.add_argument("--prepare-datasets-only", action="store_true",
                        help="Only prepare and upload datasets, don't train models")
    parser.add_argument("--force-upload", action="store_true",
                        help="Force upload datasets even if they already exist on the Hub")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Filter configurations based on command-line arguments
    configs_to_run = []
    
    if args.task_type in ["similar", "all"]:
        if args.lora_rank is not None:
            # Filter by specific LoRA rank
            configs_to_run.extend([
                config for config in SIMILAR_TASK_CONFIGS 
                if (config.use_lora and config.lora_rank == args.lora_rank) or 
                   (not config.use_lora and args.full_ft)
            ])
        elif args.full_ft:
            # Only full fine-tuning
            configs_to_run.extend([
                config for config in SIMILAR_TASK_CONFIGS 
                if not config.use_lora
            ])
        else:
            # All similar task configs
            configs_to_run.extend(SIMILAR_TASK_CONFIGS)
    
    if args.task_type in ["dissimilar", "all"]:
        if args.lora_rank is not None:
            # Filter by specific LoRA rank
            configs_to_run.extend([
                config for config in DISSIMILAR_TASK_CONFIGS 
                if (config.use_lora and config.lora_rank == args.lora_rank) or 
                   (not config.use_lora and args.full_ft)
            ])
        elif args.full_ft:
            # Only full fine-tuning
            configs_to_run.extend([
                config for config in DISSIMILAR_TASK_CONFIGS 
                if not config.use_lora
            ])
        else:
            # All dissimilar task configs
            configs_to_run.extend(DISSIMILAR_TASK_CONFIGS)
    
    # If only preparing datasets, prepare and upload them
    if args.prepare_datasets_only:
        if args.task_type in ["similar", "all"]:
            similar_tasks = SIMILAR_TASK_CONFIGS[0].tasks
            logger.info(f"Preparing similar task datasets: {similar_tasks}")
            prepare_and_upload_datasets(similar_tasks, args.force_upload)
        
        if args.task_type in ["dissimilar", "all"]:
            dissimilar_tasks = DISSIMILAR_TASK_CONFIGS[0].tasks
            logger.info(f"Preparing dissimilar task datasets: {dissimilar_tasks}")
            prepare_and_upload_datasets(dissimilar_tasks, args.force_upload)
        
        logger.info("Dataset preparation complete. Exiting without training.")
        return
    
    # Run selected configurations
    for config in configs_to_run:
        logger.info(f"Running configuration: {get_config_name(config)}")
        run_training_configuration(config)
    
    # Upload models to Hugging Face Hub if requested
    if args.upload:
        for config in configs_to_run:
            model_name = get_config_name(config)
            if os.path.exists(os.path.join(OUTPUT_DIR, model_name)):
                upload_model_to_hub(
                    model_name=model_name,
                    tasks=config.tasks,
                    lora_rank=config.lora_rank if config.use_lora else None
                )

if __name__ == "__main__":
    main()