"""
Evaluation and upload script

This script handles:
1. Model evaluation on validation sets
2. Uploading models to Hugging Face Hub
3. Uploading datasets to Hugging Face Hub
"""

import os
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from datasets import load_from_disk
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
from huggingface_hub import HfApi
from config import (
    DATA_DIR,
    MODEL_DIR,
    MODEL_NAME,
    TOKENIZER_PATH,
    HF_USERNAME,
    HF_TOKEN,
    TASK_LABELS,
    METRIC_MAPPING,
    RESULTS_FILE,
)

# Dataset descriptions for documentation
from dataset_descriptions import DATASET_DESCRIPTIONS

def compute_metrics(eval_pred, task):
    """Compute classification metrics for fine-tuned LLaMA models.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        task: Name of the task
        
    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred

    if labels is None or len(labels) == 0:
        raise ValueError("❌ Labels are missing in the dataset. Check preprocessing.")

    # Ensure labels are single integer values
    if isinstance(labels[0], list):
        labels = [label[0] for label in labels]

    predictions = torch.tensor(predictions)

    # Extract only first token logits for classification
    if predictions.dim() == 3:
        predictions = predictions[:, 0, :]

    # Ensure batch size alignment
    if predictions.shape[0] != len(labels):
        raise ValueError(
            f"❌ Mismatch: Predictions batch ({predictions.shape[0]}) "
            f"vs Labels batch ({len(labels)})."
        )

    predicted_labels = torch.argmax(predictions, dim=-1)

    # Load and compute metrics
    metric = evaluate.load("glue", task) if task in METRIC_MAPPING else evaluate.load(task)
    return metric.compute(predictions=predicted_labels, references=labels)

def evaluate_model(model_path, task):
    """Load and evaluate a trained model on validation set(s).
    
    Args:
        model_path: Path to the model
        task: Name of the task
    """
    print(f"🔍 Evaluating {model_path} on {task}...")

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=TASK_LABELS[task]
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Load dataset
    dataset_path = os.path.join(DATA_DIR, task.replace("/", "_"))
    dataset = load_from_disk(dataset_path)

    # Determine evaluation splits
    eval_splits = ["validation"] if "validation" in dataset else []
    if task == "mnli":
        eval_splits = ["validation_matched", "validation_mismatched"]

    for split in eval_splits:
        print(f"📊 Evaluating {task} on {split}...")
        eval_dataset = dataset[split]

        # Ensure dataset contains labels
        if "labels" not in eval_dataset.features:
            raise ValueError(
                f"❌ No 'labels' found in {split} for {task}. "
                "Check dataset preprocessing."
            )

        # Define evaluation settings
        training_args = TrainingArguments(
            output_dir=model_path,
            per_device_eval_batch_size=8,
            eval_accumulation_steps=10,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=lambda eval_pred: compute_metrics(eval_pred, task),
        )

        # Run evaluation
        results = trainer.evaluate()

        # Save results
        with open(RESULTS_FILE, "a") as f:
            f.write(f"{model_path} - {task} ({split}):\n")
            f.write(str(results) + "\n\n")

        print(f"✅ Results ({split}): {results}\n")

def upload_model_to_hub(model_name, task, lora_rank):
    """Upload fine-tuned models to Hugging Face Hub.
    
    Args:
        model_name: Name of the model
        task: Name of the task
        lora_rank: LoRA rank used in training
    """
    model_path = os.path.join(MODEL_DIR, model_name)
    repo_id = f"{HF_USERNAME}/LLaMA_1B_{model_name}"

    print(f"📤 Uploading {model_name} to {repo_id}...")

    # Create repository if it doesn't exist
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)

    # Load model & tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # Save model to HF Hub
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)

    # Add model card
    model_card = f"""# {model_name}

Fine-tuned LLaMA model on {task.upper()} dataset.

- **LoRA**: {'Enabled' if 'LoRA' in model_name else 'Full Fine-Tuning'}
- **LoRA Rank**: {lora_rank if 'LoRA' in model_name else 'N/A'}
- **Tasks**: {task.upper()}
- **Base Model**: LLaMA 1B
- **Optimizer**: AdamW
- **Batch Size**: 32

Trained using the 🤗 Transformers `Trainer` API.
"""
    with open(os.path.join(model_path, "README.md"), "w") as f:
        f.write(model_card)

    api.upload_folder(folder_path=model_path, repo_id=repo_id)
    print(f"Successfully uploaded {model_name} to {repo_id}.\n")

def upload_dataset(dataset_key):
    """Upload preprocessed dataset to Hugging Face Hub.
    
    Args:
        dataset_key: Key of the dataset to upload
    """
    dataset_path = os.path.join(DATA_DIR, dataset_key)

    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_key} not found at {dataset_path}, skipping...")
        return

    dataset_name = f"{dataset_key}_llama1b_modified"
    print(f"Uploading {dataset_name} to Hugging Face Hub...")

    # Load dataset from disk
    dataset = load_from_disk(dataset_path)

    # Create repository on Hugging Face
    api = HfApi()
    repo_id = f"{HF_USERNAME}/{dataset_name}"
    api.create_repo(repo_id, exist_ok=True, token=HF_TOKEN)

    # Push dataset to Hub
    dataset.push_to_hub(repo_id, token=HF_TOKEN)

    # Upload dataset card
    readme_path = os.path.join(dataset_path, "README.md")
    with open(readme_path, "w") as f:
        f.write(DATASET_DESCRIPTIONS[dataset_key])

    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        token=HF_TOKEN
    )

    print(f"Successfully uploaded {dataset_name} to {repo_id}.")

def main():
    """Main function to evaluate models and upload to Hugging Face Hub."""
    # Evaluate all trained models
    for model_name in os.listdir(MODEL_DIR):
        print(f"🔍 Evaluating {model_name}...")
        for task in METRIC_MAPPING.keys():
            if task in model_name:
                model_path = os.path.join(MODEL_DIR, model_name)
                evaluate_model(model_path, task)

    print("✅ Evaluation complete! Results saved to", RESULTS_FILE)

    # Upload models to HF Hub
    for model_name in os.listdir(MODEL_DIR):
        if "sst2_mnli_qqp_LoRA_" in model_name:
            print(f"📤 Uploading {model_name}...")
            lora_rank = int(model_name.split("_")[-1].split(".")[0])
            print(f"LoRA Rank: {lora_rank}")
            for task in METRIC_MAPPING.keys():
                if task in model_name:
                    upload_model_to_hub(model_name, task, lora_rank)

    print("All models uploaded successfully.")

    # Upload datasets to HF Hub
    for dataset_key in DATASET_DESCRIPTIONS.keys():
        if not any(task in dataset_key for task in ["sst2", "mnli", "qqp"]):
            print(f"Uploading {dataset_key}...")
            upload_dataset(dataset_key)

    print("All datasets uploaded successfully.")

if __name__ == "__main__":
    main()
