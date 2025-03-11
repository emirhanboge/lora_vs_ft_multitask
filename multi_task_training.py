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
    """Trainer for dissimilar tasks."""
    
    def get_model(self):
        """Get the model for dissimilar tasks."""
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
        """Load the combined dataset from Hugging Face Hub."""
        combined_dataset_name = self._get_combined_dataset_name()
        
        # Check if the dataset exists on the Hub
        logger.info(f"Loading combined dataset from Hugging Face Hub: {combined_dataset_name}")
        
        try:
            # Load the dataset directly from the Hub
            dataset = load_dataset(f"emirhanboge/{combined_dataset_name}")
            
            # Check if the dataset has validation and test splits
            has_validation = "validation" in dataset
            has_test = "test" in dataset
            
            if has_validation and has_test:
                logger.info("Using existing train, validation, and test splits from the Hub")
                train_dataset = dataset["train"]
                val_dataset = dataset["validation"]
                test_dataset = dataset["test"]
            elif has_validation:
                logger.info("Using existing train and validation splits from the Hub (no test split found)")
                train_dataset = dataset["train"]
                val_dataset = dataset["validation"]
                # Create test split from validation
                val_size = len(val_dataset)
                test_size = val_size // 2
                shuffled_val = val_dataset.shuffle(seed=42)
                val_dataset = shuffled_val.select(range(val_size - test_size))
                test_dataset = shuffled_val.select(range(val_size - test_size, val_size))
            else:
                # If no validation or test splits, create them from the train split
                logger.info("No validation or test splits found, creating them from the train split")
                train_dataset = dataset["train"]
                
                # Calculate sizes for train, validation, and test
                total_size = len(train_dataset)
                test_size = min(int(total_size * 0.1), 5000)  # 10% or max 5000 examples for test
                val_size = min(int(total_size * 0.1), 5000)   # 10% or max 5000 examples for validation
                train_size = total_size - val_size - test_size
                
                logger.info(f"Total dataset size: {total_size}, using {train_size} for training, {val_size} for validation, and {test_size} for testing")
                
                # Shuffle the dataset with a fixed seed
                shuffled_dataset = train_dataset.shuffle(seed=42)
                
                # Create train, validation, and test splits
                train_dataset = shuffled_dataset.select(range(train_size))
                val_dataset = shuffled_dataset.select(range(train_size, train_size + val_size))
                test_dataset = shuffled_dataset.select(range(train_size + val_size, total_size))
            
            # Sample EXAMPLES_PER_TASK examples for each task
            sampled_train = []
            sampled_val = []
            
            for task in self.config.tasks:
                # Filter train dataset for current task
                task_train = train_dataset.filter(lambda x: x["task"] == task)
                if len(task_train) > EXAMPLES_PER_TASK:
                    # Randomly sample EXAMPLES_PER_TASK examples
                    task_train = task_train.shuffle(seed=42).select(range(EXAMPLES_PER_TASK))
                sampled_train.append(task_train)
                
                # Filter validation dataset for current task
                task_val = val_dataset.filter(lambda x: x["task"] == task)
                # For validation, take 10% of EXAMPLES_PER_TASK or all examples if less
                val_size = min(len(task_val), EXAMPLES_PER_TASK // 10)
                if len(task_val) > val_size:
                    task_val = task_val.shuffle(seed=123).select(range(val_size))
                sampled_val.append(task_val)
            
            # Combine sampled datasets
            self.train_dataset = concatenate_datasets(sampled_train)
            self.eval_dataset = concatenate_datasets(sampled_val)
            
            # Process datasets to ensure proper format for training
            self.train_dataset = self._process_dataset_for_training(self.train_dataset)
            self.eval_dataset = self._process_dataset_for_training(self.eval_dataset)
            
            logger.info(f"Successfully loaded dataset from Hub: {combined_dataset_name}")
            logger.info(f"Train dataset size: {len(self.train_dataset)}, Validation dataset size: {len(self.eval_dataset)}")
            
            return
        except Exception as e:
            logger.error(f"Error loading dataset from Hub: {str(e)}")
            logger.error("Please check that the dataset exists on the Hub")
            raise
    
    def _process_dataset_for_training(self, dataset):
        """Process dataset to ensure proper format for causal language modeling."""
        
        def process_example(example):
            # Get input_ids and labels
            input_ids = example["input_ids"]
            labels = example["labels"]
            attention_mask = example["attention_mask"]
            
            # For causal LM, we need to create a single sequence
            # Create combined input_ids by concatenating input_ids and labels
            combined_input_ids = input_ids + labels
            
            # Create combined attention_mask
            combined_attention_mask = attention_mask + [1] * len(labels)
            
            # Create labels with -100 for input tokens
            combined_labels = [-100] * len(input_ids) + labels
            
            # Make sure all sequences have the same length
            max_length = min(len(combined_input_ids), 2048)  # Limit to 2048 tokens
            
            if len(combined_input_ids) > max_length:
                combined_input_ids = combined_input_ids[:max_length]
                combined_attention_mask = combined_attention_mask[:max_length]
                # Ensure labels are properly aligned
                if len(input_ids) < max_length:
                    # Some of the label tokens are included
                    label_portion = max_length - len(input_ids)
                    combined_labels = [-100] * len(input_ids) + labels[:label_portion]
                else:
                    # All tokens are from input_ids
                    combined_labels = [-100] * max_length
            
            return {
                "input_ids": combined_input_ids,
                "attention_mask": combined_attention_mask,
                "labels": combined_labels,
                "task": example["task"]
            }
        
        # Apply processing to each example
        processed_dataset = dataset.map(
            process_example,
            desc="Processing dataset for training"
        )
        
        return processed_dataset
    
    def _get_combined_dataset_name(self) -> str:
        """Get the name of the combined dataset."""
        return "_".join(self.config.tasks) + "_llama1b_modified"
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        logits, labels = eval_pred
        
        # Shift logits and labels for loss calculation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Calculate loss
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {"loss": loss.item()}
    
    def train(self):
        """Train the model on dissimilar tasks."""
        if model_already_trained(get_config_name(self.config)):
            logger.info(f"Model {get_config_name(self.config)} already trained, skipping...")
            return
        
        logger.info(f"Training model on tasks: {self.config.tasks}")
        
        # Get model and tokenizer
        model = self.get_model()
        
        # Load dataset
        self.load_dataset()
        
        # Create a custom data collator that ensures proper padding
        def custom_data_collator(features):
            # Convert list of dicts to dict of lists
            batch = {
                k: [torch.tensor(f[k]) for f in features]
                for k in features[0].keys() if k != "task"
            }
            
            # Get max length in the batch
            max_length = max(len(ids) for ids in batch["input_ids"])
            
            # Pad all tensors to max_length
            for key in ["input_ids", "attention_mask", "labels"]:
                batch[key] = [
                    torch.nn.functional.pad(
                        tensor, 
                        (0, max_length - len(tensor)), 
                        value=0 if key != "labels" else -100
                    ) 
                    for tensor in batch[key]
                ]
            
            # Stack tensors
            batch = {k: torch.stack(v) for k, v in batch.items()}
            
            # Add task if it exists
            if "task" in features[0]:
                batch["task"] = [f["task"] for f in features]
            
            return batch
        
        # Set up training arguments
        training_args = self.get_training_args()
        
        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=custom_data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        model_name = get_config_name(self.config)
        trainer.save_model(model_name)
        
        # Upload model to Hub
        upload_model_to_hub(model_name, self.config.tasks, self.config.lora_rank if self.config.use_lora else None)

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
    """Prepare and upload datasets for all tasks."""
    logger.info("Preparing datasets for tasks: {}".format(tasks))
    
    # Process each task
    for task in tasks:
        if task == "squad_v2":
            # Load SQuAD v2 dataset
            dataset = load_dataset("squad_v2")
            
            # Process dataset
            processed_dataset = process_squad_v2(dataset)
            
            # Upload to Hub
            upload_dataset_to_hub(f"{task}_llama1b", processed_dataset, force_upload)
        
        elif task == "codex_glue":
            # Load CodeXGLUE dataset
            dataset = load_dataset("code_x_glue_ct_code_to_text", "java")
            
            # Process dataset
            processed_dataset = process_codex_glue(dataset)
            
            # Upload to Hub
            upload_dataset_to_hub(f"{task}_llama1b", processed_dataset, force_upload)
        
        elif task == "cnn_dailymail":
            # Load CNN/DailyMail dataset
            dataset = load_dataset("cnn_dailymail", "3.0.0")
            
            # Process dataset
            processed_dataset = process_cnn_dailymail(dataset)
            
            # Upload to Hub
            upload_dataset_to_hub(f"{task}_llama1b", processed_dataset, force_upload)
    
    # Create combined dataset for dissimilar tasks
    if all(task in tasks for task in ["squad_v2", "codex_glue", "cnn_dailymail"]):
        create_combined_dataset(tasks, force_upload)

def process_squad_v2(dataset):
    """Process SQuAD v2 dataset for LLaMA model."""
    tokenizer = setup_tokenizer("meta-llama/Llama-3.2-1B")
    
    def format_squad(example):
        context = example["context"]
        question = example["question"]
        
        if example["answers"]["text"]:
            answer = example["answers"]["text"][0]
        else:
            answer = "No answer available in the text."
        
        input_text = f"Task: squad_v2 | Read the passage and answer. Context: {context} Question: {question}"
        
        return {
            "input_text": input_text,
            "label": answer,
            "task": "squad_v2"
        }
    
    # Process train, validation, and test sets
    train_dataset = dataset["train"].map(format_squad, remove_columns=dataset["train"].column_names)
    validation_dataset = dataset["validation"].map(format_squad, remove_columns=dataset["validation"].column_names)
    
    # Create test set from validation (since SQuAD doesn't have a test split)
    # Use 50% of validation for test
    val_size = len(validation_dataset)
    test_size = val_size // 2
    
    # Shuffle validation and split into val and test
    shuffled_val = validation_dataset.shuffle(seed=42)
    new_validation = shuffled_val.select(range(val_size - test_size))
    test_dataset = shuffled_val.select(range(val_size - test_size, val_size))
    
    logger.info(f"SQuAD v2: Train size: {len(train_dataset)}, Val size: {len(new_validation)}, Test size: {len(test_dataset)}")
    
    # Tokenize datasets
    def tokenize(examples):
        inputs = tokenizer(examples["input_text"], padding=False, truncation=True, max_length=512)
        labels = tokenizer(examples["label"], padding=False, truncation=True, max_length=128)
        
        result = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels["input_ids"],
            "task": examples["task"],
            "raw_inputs": examples["input_text"],
            "raw_labels": examples["label"]
        }
        return result
    
    tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
    tokenized_validation = new_validation.map(tokenize, batched=True, remove_columns=new_validation.column_names)
    tokenized_test = test_dataset.map(tokenize, batched=True, remove_columns=test_dataset.column_names)
    
    # Combine into a single dataset
    processed_dataset = DatasetDict({
        "train": tokenized_train,
        "validation": tokenized_validation,
        "test": tokenized_test
    })
    
    return processed_dataset

def process_codex_glue(dataset):
    """Process CodeXGLUE dataset for LLaMA model."""
    tokenizer = setup_tokenizer("meta-llama/Llama-3.2-1B")
    
    def format_codex(example):
        code = example["code"]
        docstring = example["docstring"]
        
        input_text = f"Task: codex_glue | Generate documentation for the following code: {code}"
        
        return {
            "input_text": input_text,
            "label": docstring,
            "task": "codex_glue"
        }
    
    # Process train, validation, and test sets
    train_dataset = dataset["train"].map(format_codex, remove_columns=dataset["train"].column_names)
    validation_dataset = dataset["validation"].map(format_codex, remove_columns=dataset["validation"].column_names)
    
    # Create test set from validation if test split doesn't exist
    if "test" in dataset:
        test_dataset = dataset["test"].map(format_codex, remove_columns=dataset["test"].column_names)
    else:
        # Use 50% of validation for test
        val_size = len(validation_dataset)
        test_size = val_size // 2
        
        # Shuffle validation and split into val and test
        shuffled_val = validation_dataset.shuffle(seed=42)
        validation_dataset = shuffled_val.select(range(val_size - test_size))
        test_dataset = shuffled_val.select(range(val_size - test_size, val_size))
    
    logger.info(f"CodeXGLUE: Train size: {len(train_dataset)}, Val size: {len(validation_dataset)}, Test size: {len(test_dataset)}")
    
    # Tokenize datasets
    def tokenize(examples):
        inputs = tokenizer(examples["input_text"], padding=False, truncation=True, max_length=512)
        labels = tokenizer(examples["label"], padding=False, truncation=True, max_length=128)
        
        result = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels["input_ids"],
            "task": examples["task"],
            "raw_inputs": examples["input_text"],
            "raw_labels": examples["label"]
        }
        return result
    
    tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
    tokenized_validation = validation_dataset.map(tokenize, batched=True, remove_columns=validation_dataset.column_names)
    tokenized_test = test_dataset.map(tokenize, batched=True, remove_columns=test_dataset.column_names)
    
    # Combine into a single dataset
    processed_dataset = DatasetDict({
        "train": tokenized_train,
        "validation": tokenized_validation,
        "test": tokenized_test
    })
    
    return processed_dataset

def process_cnn_dailymail(dataset):
    """Process CNN/DailyMail dataset for LLaMA model."""
    tokenizer = setup_tokenizer("meta-llama/Llama-3.2-1B")
    
    def format_cnn(example):
        article = example["article"]
        highlights = example["highlights"]
        
        input_text = f"Task: cnn_dailymail | Summarize the following article: {article}"
        
        return {
            "input_text": input_text,
            "label": highlights,
            "task": "cnn_dailymail"
        }
    
    # Process train, validation, and test sets
    train_dataset = dataset["train"].map(format_cnn, remove_columns=dataset["train"].column_names)
    validation_dataset = dataset["validation"].map(format_cnn, remove_columns=dataset["validation"].column_names)
    test_dataset = dataset["test"].map(format_cnn, remove_columns=dataset["test"].column_names)
    
    logger.info(f"CNN/DailyMail: Train size: {len(train_dataset)}, Val size: {len(validation_dataset)}, Test size: {len(test_dataset)}")
    
    # Tokenize datasets
    def tokenize(examples):
        inputs = tokenizer(examples["input_text"], padding=False, truncation=True, max_length=512)
        labels = tokenizer(examples["label"], padding=False, truncation=True, max_length=128)
        
        result = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels["input_ids"],
            "task": examples["task"],
            "raw_inputs": examples["input_text"],
            "raw_labels": examples["label"]
        }
        return result
    
    tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
    tokenized_validation = validation_dataset.map(tokenize, batched=True, remove_columns=validation_dataset.column_names)
    tokenized_test = test_dataset.map(tokenize, batched=True, remove_columns=test_dataset.column_names)
    
    # Combine into a single dataset
    processed_dataset = DatasetDict({
        "train": tokenized_train,
        "validation": tokenized_validation,
        "test": tokenized_test
    })
    
    return processed_dataset

def create_combined_dataset(tasks, force_upload=False):
    """Create a combined dataset from individual task datasets."""
    combined_train_datasets = []
    combined_val_datasets = []
    combined_test_datasets = []
    
    for task in tasks:
        try:
            # Load the individual dataset from Hub
            dataset_name = f"{HF_USERNAME}/{task}_llama1b"
            dataset = load_dataset(dataset_name)
            
            # Add train, validation, and test splits
            combined_train_datasets.append(dataset["train"])
            combined_val_datasets.append(dataset["validation"])
            combined_test_datasets.append(dataset["test"])
            logger.info(f"Loaded {task} dataset from Hub with train, validation, and test splits")
        except Exception as e:
            logger.error(f"Error loading {task} dataset: {str(e)}")
            return
    
    # Combine datasets
    combined_train = concatenate_datasets(combined_train_datasets)
    combined_val = concatenate_datasets(combined_val_datasets)
    combined_test = concatenate_datasets(combined_test_datasets)
    
    # Create a combined dataset with train, validation, and test splits
    combined_dataset = DatasetDict({
        "train": combined_train,
        "validation": combined_val,
        "test": combined_test
    })
    
    # Upload to Hub
    combined_name = "_".join(tasks) + "_llama1b_modified"
    upload_dataset_to_hub(combined_name, combined_dataset, force_upload)

def upload_dataset_to_hub(dataset_name, dataset, force_upload=False):
    """Upload a dataset to the Hugging Face Hub."""
    full_name = f"{HF_USERNAME}/{dataset_name}"
    
    # Check if dataset already exists
    if check_dataset_on_hub(full_name) and not force_upload:
        logger.info(f"Dataset {full_name} already exists on Hub, skipping upload")
        return
    
    logger.info(f"Uploading dataset {full_name} to Hub")
    
    try:
        dataset.push_to_hub(full_name)
        logger.info(f"Successfully uploaded dataset {full_name} to Hub")
    except Exception as e:
        logger.error(f"Error uploading dataset to Hub: {str(e)}")
        raise

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