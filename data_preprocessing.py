"""
Data preprocessing script

This script handles the preprocessing of datasets for both similar and dissimilar tasks.
It downloads datasets from Hugging Face, tokenizes them, and saves them to disk.
"""

import os
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from config import (
    DATA_DIR,
    MODEL_NAME,
    TOKENIZER_PATH,
    SIMILAR_TASKS,
    DISSIMILAR_TASKS,
    FORGETTING_BENCHMARKS,
    MAX_LENGTHS,
)

def setup_tokenizer():
    """Initialize and configure the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use EOS as padding
    return tokenizer

def preprocess_function(examples, task, tokenizer):
    """Tokenize and format datasets dynamically with correct labels.
    
    Args:
        examples: Dataset examples to process
        task: Name of the task
        tokenizer: Tokenizer to use
        
    Returns:
        Tokenized and formatted dataset
    """
    task_prefix = f"Task: {task} | "

    if task in SIMILAR_TASKS:  # Classification tasks (e.g., SST2, MNLI, QQP)
        if task == "mnli":
            input_texts = [
                f"{task_prefix} Premise: {p} Hypothesis: {h}" 
                for p, h in zip(examples["premise"], examples["hypothesis"])
            ]
        elif task == "qqp":
            input_texts = [
                f"{task_prefix} Q1: {q1} Q2: {q2}" 
                for q1, q2 in zip(examples["question1"], examples["question2"])
            ]
        else:
            input_texts = [
                f"{task_prefix} Sentence: {s}" 
                for s in examples["sentence"]
            ]
        labels = examples["label"] if "label" in examples else examples["labels"]

    elif task in DISSIMILAR_TASKS:  # Generation tasks (QA, Code, Summarization)
        if task == "rajpurkar/squad_v2":  # Question Answering
            input_texts = [
                f"{task_prefix} Read the passage and answer. Context: {c} Question: {q}"
                for c, q in zip(examples["context"], examples["question"])
            ]
            # Extract first answer or empty
            labels = [a["text"][0] if a["text"] else "" for a in examples["answers"]]

        elif task in ["code_to_text", "codex_glue"]:  # Code Generation
            input_texts = [
                f"{task_prefix} Generate a description for the following code snippet:\n{code}"
                for code in examples["code"]
            ]
            labels = examples["docstring"]  # Text description of code

        elif task == "cnn_dailymail":  # Summarization
            input_texts = [
                f"{task_prefix} Summarize this news article:\n{article}"
                for article in examples["article"]
            ]
            labels = examples["highlights"]  # The summary (ground-truth)

    else:  # Other tasks
        input_texts = [f"{task_prefix} {text}" for text in examples["text"]]
        labels = examples["label"] if "label" in examples else None

    # Tokenize inputs
    max_length = MAX_LENGTHS.get(task, 512)
    tokenized_output = tokenizer(
        input_texts, 
        truncation=True, 
        padding="max_length", 
        max_length=max_length
    )

    # Handle labels
    if labels is not None and task in SIMILAR_TASKS:
        if isinstance(labels, list) and isinstance(labels[0], list):
            labels = [label[0] for label in labels]  # Take only first label
    else:
        tokenized_output["labels"] = labels

    # Tokenize labels for sequence generation tasks
    if task in DISSIMILAR_TASKS:
        with tokenizer.as_target_tokenizer():
            tokenized_output["labels"] = tokenizer(
                labels, 
                truncation=True, 
                padding="max_length", 
                max_length=max_length
            )["input_ids"]

    # Store task name for multi-task later
    tokenized_output["task"] = [task] * len(input_texts)

    return tokenized_output

def process_dataset(dataset_name, tokenizer):
    """Process a single dataset.
    
    Args:
        dataset_name: Name of the dataset to process
        tokenizer: Tokenizer to use
        
    Returns:
        None, saves processed dataset to disk
    """
    dataset_path = os.path.join(DATA_DIR, dataset_name.replace("/", "_"))

    if os.path.exists(dataset_path):
        print(f"Skipping {dataset_name}, already exists at {dataset_path}", flush=True)
        return

    print(f"Processing: {dataset_name}", flush=True)

    try:
        # Load dataset from HF with appropriate config
        if dataset_name in SIMILAR_TASKS:
            dataset = load_dataset("glue", dataset_name)
        elif dataset_name == "openai_humaneval":
            dataset = load_dataset(dataset_name, trust_remote_code=True)
        elif dataset_name == "codex_glue":
            dataset = load_dataset("google/code_x_glue_ct_code_to_text", "python")
        elif dataset_name == "cnn_dailymail":
            dataset = load_dataset("cnn_dailymail", "3.0.0")
        else:
            dataset = load_dataset(dataset_name)

        print(f"Dataset {dataset_name} loaded successfully!", flush=True)

        # Tokenize dataset
        tokenized_dataset = dataset.map(
            lambda x: preprocess_function(x, dataset_name, tokenizer),
            batched=True,
        )

        # Save processed dataset
        tokenized_dataset.save_to_disk(dataset_path)
        print(f"Saved {dataset_name} to {dataset_path}\n", flush=True)

    except Exception as e:
        print(f"Error processing {dataset_name}: {e}", flush=True)

def main():
    """Main function to process all datasets."""
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)

    # Setup tokenizer
    tokenizer = setup_tokenizer()

    # Process all dissimilar tasks
    for dataset_name in DISSIMILAR_TASKS:
        process_dataset(dataset_name, tokenizer)

    # Save tokenizer if it doesn't exist
    if not os.path.exists(TOKENIZER_PATH):
        tokenizer.save_pretrained(TOKENIZER_PATH)
        print(f"✅ Tokenizer saved to {TOKENIZER_PATH}")

if __name__ == "__main__":
    main()