"""
Data preprocessing script

This script handles the preprocessing of datasets for both similar and dissimilar tasks.
It downloads datasets from Hugging Face, tokenizes them, and saves them to disk.
"""

import os
import torch
import random
import time
import logging
import json
from dataclasses import asdict
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer, 
    DataCollatorForSeq2Seq,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)
from huggingface_hub import HfApi, Repository
from typing import Optional, Dict, List, Union, Any, Tuple
from config import (
    DATA_DIR,
    MODEL_NAME,
    TOKENIZER_PATH,
    SIMILAR_TASKS,
    DISSIMILAR_TASKS,
    MAX_LENGTHS,
    TASK_LABELS,
    METRIC_MAPPING,
    DATASET_PATHS,
    EXAMPLES_PER_TASK,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseDatasetHandler:
    """Base class for dataset handling."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def validate_dataset(self, dataset: Dataset, sample_size: int = 1000) -> bool:
        """Validate dataset for potential issues."""
        return validate_dataset(dataset, self.tokenizer, sample_size)
    
    def check_dataset_on_hub(self, dataset_name: str) -> bool:
        """Check if dataset exists on hub."""
        return check_dataset_on_hub(dataset_name)
    
    def upload_model_to_hub(
        self, 
        model_name: str, 
        tasks: List[str], 
        model_path: str,
        lora_rank: Optional[int] = None,
        hf_token: Optional[str] = None,
        hf_username: Optional[str] = None
    ):
        """Upload fine-tuned models to Hugging Face Hub."""
        if not hf_username or not hf_token:
            logger.warning("HF_USERNAME or HF_TOKEN not set, skipping model upload")
            return
        
        repo_id = f"{hf_username}/{model_name}"
        logger.info(f"Uploading {model_name} to {repo_id}...")
        
        try:
            # Create repository
            api = HfApi()
            api.create_repo(repo_id, exist_ok=True, token=hf_token)
            
            # Determine model type and load
            if all(task in SIMILAR_TASKS for task in tasks):
                num_labels = len(TASK_LABELS[tasks[0]])
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, 
                    num_labels=num_labels
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path)
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Push to hub
            model.push_to_hub(repo_id, token=hf_token)
            tokenizer.push_to_hub(repo_id, token=hf_token)
            
            # Create model card
            task_names = [t.split("/")[-1] for t in tasks]
            model_card = f"""# {model_name}

Fine-tuned LLaMA model on {', '.join(task_names)} {'dataset' if len(tasks) == 1 else 'datasets'}.

•⁠  ⁠*LoRA*: {'Enabled' if 'LoRA' in model_name else 'Full Fine-Tuning'}
•⁠  ⁠*LoRA Rank*: {lora_rank if lora_rank else 'N/A'}
•⁠  ⁠*Tasks*: {', '.join(task_names)}
•⁠  ⁠*Base Model*: LLaMA 1B
•⁠  ⁠*Model Type*: {'Classification' if all(task in SIMILAR_TASKS for task in tasks) else 'Generation'}

Trained using the 🤗 Transformers ⁠ Trainer ⁠ API.
"""
            
            with open(os.path.join(model_path, "README.md"), "w") as f:
                f.write(model_card)
            
            api.upload_file(
                path_or_fileobj=os.path.join(model_path, "README.md"),
                path_in_repo="README.md",
                repo_id=repo_id,
                token=hf_token
            )
            
            logger.info(f"Successfully uploaded {model_name} to {repo_id}")
        
        except Exception as e:
            logger.error(f"Error uploading model {model_name}: {str(e)}")

class SimilarTaskDatasetHandler(BaseDatasetHandler):
    """Handler for similar task datasets (classification)."""
    
    def load_and_preprocess(self, tasks: List[str]) -> Tuple[Dataset, Dataset]:
        """Load and preprocess similar task datasets."""
        return load_and_preprocess_similar_tasks(tasks, self.tokenizer)
    
    def prepare_dataset(self, dataset_name: str):
        """Prepare a single similar task dataset."""
        dataset_path = os.path.join(DATA_DIR, dataset_name.replace("/", "_"))
        
        if os.path.exists(dataset_path):
            logger.info(f"Dataset {dataset_name} already exists at {dataset_path}")
            return
        
        try:
            dataset = load_dataset("glue", dataset_name)
            tokenized_dataset = dataset.map(
                lambda x: self._preprocess_function(x, dataset_name),
                batched=True,
                remove_columns=dataset["train"].column_names
            )
            tokenized_dataset.save_to_disk(dataset_path)
            logger.info(f"Saved {dataset_name} to {dataset_path}")
        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {e}")
    
    def _preprocess_function(self, examples: Dict, task: str) -> Dict:
        """Preprocess function for similar tasks."""
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
        tokenized["task"] = [task] * len(texts)
        return tokenized

class DissimilarTaskDatasetHandler(BaseDatasetHandler):
    """Handler for dissimilar task datasets (generation)."""
    
    def load_and_preprocess(self, tasks: List[str]) -> Tuple[Dataset, Dataset]:
        """Load and preprocess dissimilar task datasets."""
        return load_and_preprocess_dissimilar_tasks(tasks, self.tokenizer)
    
    def prepare_dataset(self, dataset_name: str):
        """Prepare a single dissimilar task dataset."""
        dataset_path = os.path.join(DATA_DIR, dataset_name.replace("/", "_"))
        
        if os.path.exists(dataset_path):
            logger.info(f"Dataset {dataset_name} already exists at {dataset_path}")
            return
        
        try:
            if dataset_name == "codex_glue":
                dataset = load_dataset("google/code_x_glue_ct_code_to_text", "python")
            elif dataset_name == "cnn_dailymail":
                dataset = load_dataset("cnn_dailymail", "3.0.0")
            else:
                dataset = load_dataset(dataset_name)
            
            tokenized_dataset = dataset.map(
                lambda x: self._preprocess_function(x, dataset_name),
                batched=True,
                remove_columns=dataset["train"].column_names
            )
            tokenized_dataset.save_to_disk(dataset_path)
            logger.info(f"Saved {dataset_name} to {dataset_path}")
        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {e}")
    
    def _preprocess_function(self, examples: Dict, task: str) -> Dict:
        """Preprocess function for dissimilar tasks."""
        if task == "squad_v2":
            input_texts = [
                f"Read the passage and answer. Context: {c} Question: {q}"
                for c, q in zip(examples["context"], examples["question"])
            ]
            labels = [a["text"][0] if a["text"] else "" for a in examples["answers"]]
        elif task in ["code_to_text", "codex_glue"]:
            input_texts = [
                f"Generate a description for the following code snippet:\n{code}"
                for code in examples["code"]
            ]
            labels = examples["docstring"]
        elif task == "cnn_dailymail":
            input_texts = [
                f"Summarize this news article:\n{article}"
                for article in examples["article"]
            ]
            labels = examples["highlights"]
        else:
            input_texts = examples["text"]
            labels = examples.get("label", None)
        
        tokenized = self.tokenizer(
            input_texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTHS[task],
            return_tensors="pt",
        )
        
        if labels is not None:
            with self.tokenizer.as_target_tokenizer():
                tokenized["labels"] = self.tokenizer(
                    labels,
                    truncation=True,
                    padding="max_length",
                    max_length=MAX_LENGTHS[task]
                )["input_ids"]
        
        tokenized["task"] = [task] * len(input_texts)
        return tokenized

def setup_tokenizer(model_name: str):
    """Initialize and configure the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def check_dataset_on_hub(dataset_name: str) -> bool:
    """Check if a dataset exists on the Hugging Face Hub."""
    try:
        api = HfApi()
        if "/" not in dataset_name:
            dataset_name = f"emirhanboge/{dataset_name}"
        api.dataset_info(dataset_name)
        return True
    except Exception:
        return False

def validate_dataset(dataset, tokenizer, sample_size=1000):
    """Validate the dataset for potential issues."""
    start_time = time.time()
    dataset_size = len(dataset)
    
    if sample_size and sample_size < dataset_size:
        logger.info(f"Validating a random sample of {sample_size} examples")
        indices = random.sample(range(dataset_size), sample_size)
    else:
        logger.info(f"Validating all {dataset_size} examples")
        indices = range(dataset_size)
    
    try:
        batch_size = 100
        num_batches = (len(indices) + batch_size - 1) // batch_size
        negative_ids_count = 0
        examples_with_negative_ids = []
        
        for i in range(num_batches):
            batch_indices = indices[i * batch_size:(i + 1) * batch_size]
            batch = dataset.select(batch_indices)
            
            for idx, example in enumerate(batch):
                if 'input_ids' in example and isinstance(example['input_ids'], list):
                    neg_ids = [id for id in example['input_ids'] if id < 0]
                    if neg_ids:
                        negative_ids_count += len(neg_ids)
                        if len(examples_with_negative_ids) < 5:
                            examples_with_negative_ids.append(batch_indices[idx])
        
        if negative_ids_count > 0:
            logger.warning(f"Found {negative_ids_count} negative input_ids")
            return False
        return True
        
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        return False

def load_and_preprocess_similar_tasks(tasks: List[str], tokenizer: AutoTokenizer) -> Tuple[Dataset, Dataset]:
    """
    Load and preprocess datasets for similar tasks (classification) from the Hub.
    
    Args:
        tasks: List of task names to process
        tokenizer: The tokenizer to use for preprocessing
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    logger.info(f"Loading similar tasks from Hub: {tasks}")
    
    try:
        # Load the combined dataset from Hub
        dataset = load_dataset("emirhanboge/squad_v2_codex_glue_cnn_dailymail_llama1b_modified")
        logger.info("Successfully loaded dataset from Hub")
        
        # Split into train and validation
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
        
        # Filter for similar tasks
        train_dataset = train_dataset.filter(lambda x: x["task"] in tasks)
        eval_dataset = eval_dataset.filter(lambda x: x["task"] in tasks)
        
        # Sample exactly 25000 examples per task
        train_samples = []
        eval_samples = []
        
        for task in tasks:
            # Sample training data
            task_data = train_dataset.filter(lambda x: x["task"] == task)
            if len(task_data) > 25000:
                random.seed(42)  # Set seed for reproducibility
                train_indices = random.sample(range(len(task_data)), 25000)
                task_data = task_data.select(train_indices)
            train_samples.append(task_data)
            
            # Sample validation data - 10% of training size
            task_eval_data = eval_dataset.filter(lambda x: x["task"] == task)
            val_sample_size = 2500  # 10% of training size
            if len(task_eval_data) > val_sample_size:
                random.seed(123)  # Different seed for validation
                val_indices = random.sample(range(len(task_eval_data)), val_sample_size)
                task_eval_data = task_eval_data.select(val_indices)
            eval_samples.append(task_eval_data)
        
        # Combine datasets
        train_dataset = concatenate_datasets(train_samples)
        eval_dataset = concatenate_datasets(eval_samples)
        
        # Ensure all sequences have consistent lengths
        max_length = 512  # Standard length for classification tasks
        
        def ensure_length_consistency(example):
            # Truncate or pad input_ids and attention_mask
            input_ids = example['input_ids'][:max_length]
            attention_mask = example['attention_mask'][:max_length]
            
            # Pad if necessary
            if len(input_ids) < max_length:
                input_ids = input_ids + [tokenizer.pad_token_id] * (max_length - len(input_ids))
                attention_mask = attention_mask + [0] * (max_length - len(attention_mask))
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': example['labels'],
                'task': example['task']
            }
        
        train_dataset = train_dataset.map(ensure_length_consistency)
        eval_dataset = eval_dataset.map(ensure_length_consistency)
        
        logger.info(f"Final dataset sizes - Train: {len(train_dataset)}, Validation: {len(eval_dataset)}")
        return train_dataset, eval_dataset
        
    except Exception as e:
        logger.error(f"Error loading dataset from Hub: {e}")
        raise

def load_and_preprocess_dissimilar_tasks(tasks: List[str], tokenizer: AutoTokenizer) -> Tuple[Dataset, Dataset]:
    """
    Load and preprocess datasets for dissimilar tasks (generation) from the Hub.
    
    Args:
        tasks: List of task names to process
        tokenizer: The tokenizer to use for preprocessing
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    logger.info(f"Loading dissimilar tasks from Hub: {tasks}")
    
    try:
        # Load the combined dataset from Hub
        dataset = load_dataset("emirhanboge/squad_v2_codex_glue_cnn_dailymail_llama1b_modified")
        logger.info("Successfully loaded dataset from Hub")
        
        # Split into train and validation
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
        
        # Create task filter
        task_filter = lambda x: x["task"] in tasks
        
        # Filter and sample in one pass for training data
        logger.info("Processing training data...")
        train_indices = {}
        for task in tasks:
            task_mask = [x["task"] == task for x in train_dataset]
            task_indices = [i for i, is_task in enumerate(task_mask) if is_task]
            if len(task_indices) > 25000:
                random.seed(42)  # Set seed for reproducibility
                task_indices = random.sample(task_indices, 25000)
            train_indices[task] = task_indices
        
        # Combine all indices and select data
        all_train_indices = [idx for indices in train_indices.values() for idx in indices]
        train_dataset = train_dataset.select(all_train_indices)
        
        # Process validation data similarly but with 10% size
        logger.info("Processing validation data...")
        eval_indices = {}
        for task in tasks:
            task_mask = [x["task"] == task for x in eval_dataset]
            task_indices = [i for i, is_task in enumerate(task_mask) if is_task]
            val_sample_size = 2500  # 10% of training size
            if len(task_indices) > val_sample_size:
                random.seed(123)  # Different seed for validation
                task_indices = random.sample(task_indices, val_sample_size)
            eval_indices[task] = task_indices
        
        # Combine all validation indices and select data
        all_eval_indices = [idx for indices in eval_indices.values() for idx in indices]
        eval_dataset = eval_dataset.select(all_eval_indices)
        
        # Ensure all sequences have consistent lengths
        max_length = 2048 # Maximum length for generation tasks
        
        def ensure_length_consistency(example):
            # Truncate or pad input_ids and attention_mask
            input_ids = example['input_ids'][:max_length]
            attention_mask = example['attention_mask'][:max_length]
            labels = example['labels'][:max_length]
            
            # Pad if necessary
            if len(input_ids) < max_length:
                input_ids = input_ids + [tokenizer.pad_token_id] * (max_length - len(input_ids))
                attention_mask = attention_mask + [0] * (max_length - len(attention_mask))
                labels = labels + [-100] * (max_length - len(labels))  # Use -100 for padding in labels
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'task': example['task']
            }
        
        # Use batched processing for faster length consistency
        train_dataset = train_dataset.map(
            ensure_length_consistency,
            num_proc=4,  # Use multiple processes
            batched=True,
            batch_size=1000,  # Process in larger batches
            desc="Processing training data"
        )
        
        eval_dataset = eval_dataset.map(
            ensure_length_consistency,
            num_proc=4,  # Use multiple processes
            batched=True,
            batch_size=1000,  # Process in larger batches
            desc="Processing validation data"
        )
        
        logger.info(f"Final dataset sizes - Train: {len(train_dataset)}, Validation: {len(eval_dataset)}")
        return train_dataset, eval_dataset
        
    except Exception as e:
        logger.error(f"Error loading dataset from Hub: {e}")
        raise

def main():
    """Main function to process all datasets."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Setup tokenizer
    tokenizer = setup_tokenizer(MODEL_NAME)
    
    # Process similar tasks
    similar_handler = SimilarTaskDatasetHandler(tokenizer)
    for task in SIMILAR_TASKS:
        similar_handler.prepare_dataset(task)
    
    # Process dissimilar tasks
    dissimilar_handler = DissimilarTaskDatasetHandler(tokenizer)
    for task in DISSIMILAR_TASKS:
        dissimilar_handler.prepare_dataset(task)
    
    # Save tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        tokenizer.save_pretrained(TOKENIZER_PATH)
        logger.info(f"Tokenizer saved to {TOKENIZER_PATH}")

if __name__ == "__main__":
    main()