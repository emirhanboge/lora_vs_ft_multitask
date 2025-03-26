import torch
import torch.multiprocessing as mp
import datasets
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import evaluate
from checkpoint_utils import load_model_for_checkpoint, get_checkpoint_steps
import config
import os
import dataclasses
from typing import Dict, Optional, List, Any
import json
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false" 
seed = 42
set_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Enable half-precision computation to reduce memory usage and improve speed
torch.backends.cuda.matmul.allow_tf32 = True  # For GPUs that support TF32 (like A100)
torch.backends.cudnn.benchmark = True         # Optimize cudnn for fixed-size inputs

# Batch size for each task
BATCH_SIZE = 1
SIMILAR_TASK_CONFIGS = {
    "full_ft": {
        "name": "emirhanboge/LLaMA_1B_sst2_mnli_qqp_FullFT",
        "type": "full_ft",
        "task_type": "similar"
    },
    # "lora_4": {
    #     "name": "emirhanboge/LLaMA_1B_sst2_mnli_qqp_LoRA_4",
    #     "type": "lora",
    #     "rank": 4,
    #     "task_type": "similar"
    # },
    # "lora_8": {
    #     "name": "emirhanboge/LLaMA_1B_sst2_mnli_qqp_LoRA_8",
    #     "type": "lora",
    #     "rank": 8,
    #     "task_type": "similar"
    # },
    # "lora_16": {
    #     "name": "emirhanboge/LLaMA_1B_sst2_mnli_qqp_LoRA_16",
    #     "type": "lora",
    #     "rank": 16,
    #     "task_type": "similar"
    # },
    # "lora_32": {
    #     "name": "emirhanboge/LLaMA_1B_sst2_mnli_qqp_LoRA_32",
    #     "type": "lora",
    #     "rank": 32,
    #     "task_type": "similar"
    # },
    # "lora_64": {
    #     "name": "emirhanboge/LLaMA_1B_sst2_mnli_qqp_LoRA_64",
    #     "type": "lora",
    #     "rank": 64,
    #     "task_type": "similar"
    # }
}

# Model configurations for dissimilar tasks
DISSIMILAR_TASK_CONFIGS = {
    "full_ft": {
        "name": "emirhanboge/LLaMA_1B_qa_code_summarization_dissimilar_FullFT",
        "type": "full_ft",
        "task_type": "dissimilar"
    },
    "lora_4": {
        "name": "emirhanboge/LLaMA_1B_qa_code_summarization_dissimilar_LoRA_4",
        "type": "lora",
        "rank": 4,
        "task_type": "dissimilar"
    },
    "lora_8": {
        "name": "emirhanboge/LLaMA_1B_qa_code_summarization_dissimilar_LoRA_8",
        "type": "lora",
        "rank": 8,
        "task_type": "dissimilar"
    },
    "lora_16": {
        "name": "emirhanboge/LLaMA_1B_qa_code_summarization_dissimilar_LoRA_16",
        "type": "lora",
        "rank": 16,
        "task_type": "dissimilar"
    },
    "lora_32": {
        "name": "emirhanboge/LLaMA_1B_qa_code_summarization_dissimilar_LoRA_32",
        "type": "lora",
        "rank": 32,
        "task_type": "dissimilar"
    },
    "lora_64": {
        "name": "emirhanboge/LLaMA_1B_qa_code_summarization_dissimilar_LoRA_64",
        "type": "lora",
        "rank": 64,
        "task_type": "dissimilar"
    }
}

@dataclasses.dataclass
class EvaluationResult:
    """Data class for evaluation results"""
    model_name: str
    task: str
    rouge1: Optional[float] = None
    rouge2: Optional[float] = None
    rougeL: Optional[float] = None
    bleu: Optional[float] = None
    exact_match: Optional[float] = None
    f1: Optional[float] = None
    accuracy: Optional[float] = None

    def to_dict(self):
        """Convert results to dictionary format"""
        return {
            "model_name": self.model_name,
            "task": self.task,
            "metrics": {
                "rouge1": self.rouge1,
                "rouge2": self.rouge2,
                "rougeL": self.rougeL,
                "bleu": self.bleu,
                "exact_match": self.exact_match,
                "f1": self.f1, 
                "accuracy": self.accuracy
            }
        }

def collate_classification_batch(batch):
    """Create batches for classification tasks"""
    # Handle potential edge cases
    if len(batch) == 0:
        return {}
        
    # For single sample case, add dimensions
    if len(batch) == 1:
        input_ids = torch.tensor([batch[0]["input_ids"]], dtype=torch.long)
        attention_mask = torch.tensor([batch[0]["attention_mask"]], dtype=torch.long)
        labels = torch.tensor([batch[0]["label"]], dtype=torch.long)
    else:
        # Normal batching, ensure all sequences have the same length
        # Get maximum length
        max_len = max(len(item["input_ids"]) for item in batch)
        
        # Initialize tensors
        batch_size = len(batch)
        input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        
        # Fill tensors
        for i, item in enumerate(batch):
            seq_len = len(item["input_ids"])
            input_ids[i, :seq_len] = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask[i, :seq_len] = torch.tensor(item["attention_mask"], dtype=torch.long)
        
        # Process labels
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# Similar modifications for other collate functions if needed
def collate_generation_batch(batch):
    """Create batches for generation tasks"""
    # Get maximum lengths
    max_input_len = max(len(item["input_ids"]) for item in batch)
    max_label_len = max(len(item["labels"]) for item in batch)
    
    batch_size = len(batch)
    input_ids = torch.zeros((batch_size, max_input_len), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_input_len), dtype=torch.long)
    labels = torch.zeros((batch_size, max_label_len), dtype=torch.long)
    
    # Fill tensors
    for i, item in enumerate(batch):
        input_len = len(item["input_ids"])
        label_len = len(item["labels"])
        
        input_ids[i, :input_len] = torch.tensor(item["input_ids"], dtype=torch.long)
        attention_mask[i, :input_len] = torch.tensor(item["attention_mask"], dtype=torch.long)
        labels[i, :label_len] = torch.tensor(item["labels"], dtype=torch.long)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
def collate_qa_batch(batch):
    """Create batches for question answering tasks"""
    input_ids = torch.tensor(np.array([item["input_ids"] for item in batch]), dtype=torch.long)
    attention_mask = torch.tensor(np.array([item["attention_mask"] for item in batch]), dtype=torch.long)
    
    # For QA tasks, we also need to track question IDs and context
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "question_ids": [item["question_id"] for item in batch],
        "contexts": [item["context"] for item in batch],
        "questions": [item["question"] for item in batch],
        "answers": [item["answers"] for item in batch]
    }

def load_and_prepare_dataset(task_name, tokenizer, split="validation"):
    """Load and preprocess dataset for a specific task"""
    dataset_path = config.DATASET_PATHS.get(task_name)
    if not dataset_path:
        print(f"Warning: Dataset path not found for task {task_name}")
        return None, None

    try:
        # First try to load the complete dataset, then get the appropriate split
        dataset = load_dataset(dataset_path)
        
        # For tasks like MNLI, there might be different validation set names
        if split == "validation":
            eval_dataset = dataset.get("validation", dataset.get("validation_matched", None))
            if eval_dataset is None:
                # If no standard validation set, try using the test set
                eval_dataset = dataset.get("test", None)
        else:
            eval_dataset = dataset.get(split, None)
            
        if eval_dataset is None:
            print(f"Warning: Could not find split '{split}' for dataset {task_name}")
            # Try using the first split in the dataset
            split_names = list(dataset.keys())
            if split_names:
                eval_dataset = dataset[split_names[0]]
                print(f"Using '{split_names[0]}' split instead")
            else:
                print(f"No suitable split found for dataset {task_name}")
                return None, None
            
        print(f"Loaded dataset for {task_name}: {len(eval_dataset)} examples")
        dataset = eval_dataset
        
        # Process dataset based on task type
        if task_name in config.SIMILAR_TASKS:
            # Classification tasks
            def preprocess_classification(examples):
                if task_name == "sst2":
                    # SST2 sentiment classification
                    texts = examples["sentence"]
                    labels = examples["label"]
                    encoded = tokenizer(texts, padding="max_length", truncation=True, 
                                     max_length=config.MAX_LENGTHS.get(task_name, 128))
                    encoded["label"] = labels
                    return encoded
                elif task_name == "mnli":
                    # MNLI natural language inference
                    premise = examples["premise"]
                    hypothesis = examples["hypothesis"]
                    texts = [f"premise: {p} hypothesis: {h}" for p, h in zip(premise, hypothesis)]
                    labels = examples["label"]
                    encoded = tokenizer(texts, padding="max_length", truncation=True, 
                                     max_length=config.MAX_LENGTHS.get(task_name, 128))
                    encoded["label"] = labels
                    return encoded
                elif task_name == "qqp":
                    # QQP question pairs
                    question1 = examples["question1"]
                    question2 = examples["question2"]
                    texts = [f"question1: {q1} question2: {q2}" for q1, q2 in zip(question1, question2)]
                    labels = examples["label"]
                    encoded = tokenizer(texts, padding="max_length", truncation=True, 
                                     max_length=config.MAX_LENGTHS.get(task_name, 128))
                    encoded["label"] = labels
                    return encoded
                return {}
            
            # Process classification dataset
            processed_dataset = dataset.map(
                preprocess_classification,
                batched=True,
                remove_columns=dataset.column_names,
                load_from_cache_file=False
            )
            return processed_dataset, collate_classification_batch
            
        elif task_name == "squad_v2":
            # SQuAD question answering task
            def preprocess_squad(examples):
                questions = examples["question"]
                contexts = examples["context"]
                answers = examples["answers"]
                
                inputs = tokenizer(
                    questions, contexts,
                    padding="max_length",
                    truncation="only_second",
                    max_length=384,
                    return_token_type_ids=True,
                    return_tensors="pt"
                )
                
                # Convert inputs from dict of lists to list of dicts
                result = []
                for i in range(len(questions)):
                    item = {
                        "question_id": i,
                        "question": questions[i],
                        "context": contexts[i],
                        "answers": answers[i],
                        "input_ids": inputs["input_ids"][i].numpy(),
                        "attention_mask": inputs["attention_mask"][i].numpy(),
                        "token_type_ids": inputs["token_type_ids"][i].numpy() if "token_type_ids" in inputs else None
                    }
                    result.append(item)
                return result
                
            # Process QA dataset
            processed_dataset = dataset.map(
                preprocess_squad,
                batched=True,
                remove_columns=dataset.column_names,
                load_from_cache_file=False
            )
            return processed_dataset, collate_qa_batch
            
        elif task_name == "cnn_dailymail":
            # CNN/DailyMail summarization task
            def preprocess_summarization(examples):
                inputs = ["summarize: " + doc for doc in examples["article"]]
                model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
                
                # Process target summaries
                labels = tokenizer(examples["highlights"], max_length=128, truncation=True, padding="max_length")
                model_inputs["labels"] = labels["input_ids"]
                
                return model_inputs
                
            # Process summarization dataset
            processed_dataset = dataset.map(
                preprocess_summarization,
                batched=True,
                remove_columns=dataset.column_names,
                load_from_cache_file=False
            )
            return processed_dataset, collate_generation_batch
            
        elif task_name == "codex_glue":
            # CodeXGLUE code processing task
            def preprocess_code(examples):
                docstrings = examples["docstring"]
                code = examples["code"]
                
                inputs = ["Generate code from docstring: " + doc for doc in docstrings]
                model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
                
                # Process target code
                labels = tokenizer(code, max_length=128, truncation=True, padding="max_length")
                model_inputs["labels"] = labels["input_ids"]
                
                return model_inputs
                
            # Process code generation dataset
            processed_dataset = dataset.map(
                preprocess_code,
                batched=True,
                remove_columns=dataset.column_names,
                load_from_cache_file=False
            )
            return processed_dataset, collate_generation_batch
            
    except Exception as e:
        print(f"Error loading dataset for {task_name}: {e}")
        return None, None

    print(f"No processing method defined for task {task_name}")
    return None, None

def evaluate_classification_model(model, dataset, dataloader, device):
    """Evaluate classification model"""
    model.eval()
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating classification"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            
            logits = outputs.logits
            
            # Handle different output shapes
            if len(logits.shape) == 2:
                # Standard classification output [batch_size, num_classes]
                preds = torch.argmax(logits, dim=-1)
            else:
                # For generation model logits [batch_size, seq_len, vocab_size],
                # only consider predictions for the first token (simplified handling)
                preds = torch.argmax(logits[:, 0, :], dim=-1)
            
            predictions.extend(preds.cpu().numpy())
            references.extend(batch["labels"].cpu().numpy())
    
    # Calculate accuracy
    accuracy = sum(p == r for p, r in zip(predictions, references)) / len(predictions)
    
    # Calculate F1 score (automatically select appropriate averaging method based on classification task type)
    try:
        f1_metric = evaluate.load("f1")
        unique_classes = len(set(references))
        
        if unique_classes == 2:  # Binary classification
            f1 = f1_metric.compute(predictions=predictions, references=references, average="binary")["f1"]
        else:  # Multi-class classification
            f1 = f1_metric.compute(predictions=predictions, references=references, average="macro")["f1"]
    except Exception as e:
        print(f"Error computing F1 score: {e}")
        f1 = None
    
    return {
        "accuracy": accuracy,
        "f1": f1
    }

def evaluate_generation_model(model, dataset, dataloader, tokenizer, device, task_name):
    """Evaluate generation model"""
    model.eval()
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating generation"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # For generation tasks, use greedy decoding
            generated_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode generated sequences
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Decode reference sequences (from labels field)
            reference_texts = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
            
            predictions.extend(generated_texts)
            references.extend(reference_texts)
    
    # Calculate ROUGE scores
    try:
        rouge_metric = evaluate.load("rouge")
        rouge_results = rouge_metric.compute(predictions=predictions, references=references)
    except Exception as e:
        print(f"Error computing ROUGE scores: {e}")
        rouge_results = {"rouge1": None, "rouge2": None, "rougeL": None}
    
    # Calculate BLEU score
    try:
        bleu_metric = evaluate.load("bleu")
        bleu_result = bleu_metric.compute(predictions=predictions, references=[[r] for r in references])
    except Exception as e:
        print(f"Error computing BLEU score: {e}")
        bleu_result = {"bleu": None}
    
    return {
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
        "bleu": bleu_result["bleu"]
    }

def evaluate_qa_model(model, dataset, dataloader, tokenizer, device):
    """Evaluate question answering model"""
    model.eval()
    
    predictions = []
    references = []
    
    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating QA"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                # For QA tasks, use greedy decoding
                try:
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=50,
                        num_beams=4,
                        early_stopping=True
                    )
                    
                    # Decode generated answers
                    predicted_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    
                    # Collect predictions and reference answers
                    for i, pred_answer in enumerate(predicted_answers):
                        predictions.append(pred_answer)
                        references.append(batch["answers"][i])
                except Exception as e:
                    print(f"Error during generation for QA batch: {e}")
                    continue
        
        # If no successful predictions, return empty results
        if not predictions:
            print("No predictions generated for QA task")
            return {"exact_match": None, "f1": None}
        
        # Calculate exact match rate
        try:
            exact_match_metric = evaluate.load("exact_match")
            # Ensure references have the correct format
            ref_texts = []
            for r in references:
                if isinstance(r, dict) and "text" in r and r["text"]:
                    ref_texts.append(r["text"][0])
                elif isinstance(r, list) and r:
                    ref_texts.append(r[0])
                else:
                    ref_texts.append("")  # Use empty string as fallback
                    
            exact_match = exact_match_metric.compute(predictions=predictions, references=ref_texts)
            exact_match_value = exact_match.get("exact_match", 0)
        except Exception as e:
            print(f"Error computing exact match: {e}")
            exact_match_value = None
        
        # Calculate F1 score
        try:
            squad_metric = evaluate.load("squad")
            f1_results = []
            
            for pred, ref in zip(predictions, references):
                # Ensure reference has the correct format
                if isinstance(ref, dict) and "text" in ref:
                    ref_dict = ref
                else:
                    # Create reference in squad format
                    if isinstance(ref, list):
                        ref_text = ref[0] if ref else ""
                    else:
                        ref_text = str(ref)
                    ref_dict = {"text": [ref_text], "answer_start": [0]}
                
                try:
                    result = squad_metric.compute(
                        predictions=[{"prediction_text": pred, "id": "dummy"}],
                        references=[{"answers": ref_dict, "id": "dummy"}]
                    )
                    f1_results.append(result.get("f1", 0))
                except Exception as inner_e:
                    print(f"Error computing F1 for individual QA example: {inner_e}")
            
            f1 = sum(f1_results) / len(f1_results) if f1_results else 0
        except Exception as e:
            print(f"Error computing F1 score for QA: {e}")
            f1 = None
    except Exception as e:
        print(f"Error in QA evaluation: {e}")
        return {"exact_match": None, "f1": None}
    
    return {
        "exact_match": exact_match["exact_match"],
        "f1": f1
    }

def evaluate_model_on_task(model, tokenizer, task_name, device):
    """Evaluate model on a specific task"""
    # Load and prepare dataset
    try:
        dataset, collate_fn = load_and_prepare_dataset(task_name, tokenizer)
        if dataset is None:
            print(f"Skipping evaluation for task {task_name} - dataset not available")
            return {
                "accuracy": None,
                "f1": None,
                "rouge1": None,
                "rouge2": None,
                "rougeL": None,
                "bleu": None,
                "exact_match": None
            }
            
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Choose evaluation method based on task type
        if task_name in ["sst2", "mnli", "qqp"]:
            metrics = evaluate_classification_model(model, dataset, dataloader, device)
            # For classification tasks, only return accuracy and f1
            return {
                "accuracy": metrics.get("accuracy"),
                "f1": metrics.get("f1"),
                "rouge1": None,
                "rouge2": None,
                "rougeL": None,
                "bleu": None,
                "exact_match": None
            }
        elif task_name in ["cnn_dailymail", "codex_glue"]:
            metrics = evaluate_generation_model(model, dataset, dataloader, tokenizer, device, task_name)
            # For generation tasks, return rouge and bleu metrics
            return {
                "accuracy": None,
                "f1": None,
                "rouge1": metrics.get("rouge1"),
                "rouge2": metrics.get("rouge2"),
                "rougeL": metrics.get("rougeL"),
                "bleu": metrics.get("bleu"),
                "exact_match": None
            }
        elif task_name == "squad_v2":
            metrics = evaluate_qa_model(model, dataset, dataloader, tokenizer, device)
            # For QA tasks, return exact_match and f1
            return {
                "accuracy": None,
                "f1": metrics.get("f1"),
                "rouge1": None,
                "rouge2": None,
                "rougeL": None,
                "bleu": None,
                "exact_match": metrics.get("exact_match")
            }
        else:
            print(f"Unsupported task type: {task_name}")
            return {
                "accuracy": None,
                "f1": None,
                "rouge1": None,
                "rouge2": None,
                "rougeL": None,
                "bleu": None,
                "exact_match": None
            }
    except Exception as e:
        print(f"Error evaluating task {task_name}: {e}")
        return {
            "accuracy": None,
            "f1": None,
            "rouge1": None,
            "rouge2": None,
            "rougeL": None,
            "bleu": None,
            "exact_match": None
        }

def evaluate_checkpoint(model_config, checkpoint_step, origin_model, tokenizer, device, use_fp16=True):
    """Evaluate model at specific checkpoint"""
    model_name = model_config["name"]
    model_type = model_config["type"]
    task_type = model_config.get("task_type", "dissimilar")
    
    print(f"\n===== Evaluating checkpoint at step {checkpoint_step} =====")
    
    try:
        # First try to download and inspect the config file for this checkpoint
        from huggingface_hub import hf_hub_download
        import json
        import os
        
        try:
            print(f"Attempting to access config for checkpoint at {model_name}/checkpoint-{checkpoint_step}")
            # Get the config file for this checkpoint
            config_path = hf_hub_download(
                repo_id=model_name,
                filename="config.json",
                subfolder=f"checkpoint-{checkpoint_step}"
            )
            
            print(f"Successfully found config at: {config_path}")
            
            # Read the config to determine model architecture
            with open(config_path, 'r') as f:
                model_config_json = json.load(f)  # Renamed from config_json to avoid shadowing
            
            # Get model type info from config
            model_architecture = model_config_json.get("architectures", ["LlamaForCausalLM"])[0]
            print(f"Model architecture from config: {model_architecture}")
            
            # Fix the RoPE scaling configuration issue
            if "rope_scaling" in model_config_json:
                rope_config = model_config_json["rope_scaling"]
                if "original_max_position_embeddings" in rope_config:
                    max_pos_embeddings = model_config_json.get("max_position_embeddings", 2048)
                    if rope_config["original_max_position_embeddings"] >= max_pos_embeddings:
                        print("Fixing RoPE scaling configuration")
                        # Either remove it or fix the values
                        model_config_json.pop("rope_scaling", None)
                        # Save the modified config
                        with open(config_path, 'w') as f:
                            json.dump(model_config_json, f, indent=2)
            
            # Get the cache directory where the config file is stored
            cache_dir = os.path.dirname(os.path.dirname(config_path))
            checkpoint_dir = os.path.join(cache_dir, f"checkpoint-{checkpoint_step}")
            print(f"Using local cache path: {checkpoint_dir}")
            
            # Load the model based on config architecture
            if model_type == "lora":
                # For LoRA models
                print("Loading as LoRA model")
                original_model = AutoModelForCausalLM.from_pretrained(origin_model)
                
                # Ensure padding token is set
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    original_model.config.pad_token_id = tokenizer.pad_token_id
                
                from peft import PeftModel
                finetuned_model = PeftModel.from_pretrained(
                    original_model, 
                    checkpoint_dir,
                    trust_remote_code=True,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                model = finetuned_model.merge_and_unload()
            else:
                # For full fine-tuned models
                if "CausalLM" in model_architecture:
                    print("Loading as Causal LM model")
                    model = AutoModelForCausalLM.from_pretrained(
                        checkpoint_dir,
                        trust_remote_code=True,
                        ignore_mismatched_sizes=True,
                        device_map="auto" if torch.cuda.is_available() else None,
                        # Add extra parameters to handle RoPE scaling issues
                        rope_scaling=None  # Disable RoPE scaling
                    )
                elif "SequenceClassification" in model_architecture:
                    print("Loading as Sequence Classification model")
                    model = AutoModelForSequenceClassification.from_pretrained(
                        checkpoint_dir,
                        trust_remote_code=True,
                        ignore_mismatched_sizes=True,
                        device_map="auto" if torch.cuda.is_available() else None,
                        rope_scaling=None  # Disable RoPE scaling
                    )
                else:
                    # Default to causal LM if architecture is unclear
                    print(f"Unknown architecture: {model_architecture}, trying Causal LM")
                    model = AutoModelForCausalLM.from_pretrained(
                        checkpoint_dir,
                        trust_remote_code=True,
                        ignore_mismatched_sizes=True,
                        device_map="auto" if torch.cuda.is_available() else None,
                        rope_scaling=None  # Disable RoPE scaling
                    )
                    
        except Exception as config_error:
            print(f"Error with config approach: {config_error}")
            print("Falling back to alternative loading methods")
            
            # Try multiple model classes with RoPE scaling disabled
            if model_type == "lora":
                # LoRA models
                print("Trying alternative LoRA loading")
                original_model = AutoModelForCausalLM.from_pretrained(
                    origin_model,
                    rope_scaling=None  # Disable RoPE scaling
                )
                
                # Ensure padding token
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    original_model.config.pad_token_id = tokenizer.pad_token_id
                
                # Try with repo_id and revision format
                from peft import PeftModel
                finetuned_model = PeftModel.from_pretrained(
                    original_model,
                    model_name,
                    subfolder=f"checkpoint-{checkpoint_step}",
                    trust_remote_code=True
                )
                model = finetuned_model.merge_and_unload()
            else:
                # Try direct loading with modified config
                try:
                    print("Trying to load with modified config")
                    
                    # Download config first
                    from transformers import AutoConfig
                    try:
                        # Renamed to model_auto_config to avoid shadowing
                        model_auto_config = AutoConfig.from_pretrained(
                            f"{model_name}/checkpoint-{checkpoint_step}",
                            trust_remote_code=True
                        )
                        # Remove or fix RoPE scaling
                        if hasattr(model_auto_config, "rope_scaling"):
                            delattr(model_auto_config, "rope_scaling")
                        
                        # Now load model with the modified config
                        model = AutoModelForCausalLM.from_pretrained(
                            f"{model_name}/checkpoint-{checkpoint_step}",
                            config=model_auto_config,
                            trust_remote_code=True,
                            ignore_mismatched_sizes=True
                        )
                        print("Successfully loaded with modified config")
                    except Exception as e:
                        print(f"Failed with modified config: {e}")
                        raise
                        
                except Exception:
                    # Full fine-tuned models - try both model classes
                    for model_class in [AutoModelForCausalLM, AutoModelForSequenceClassification]:
                        try:
                            print(f"Trying to load with {model_class.__name__}")
                            model = model_class.from_pretrained(
                                model_name,
                                subfolder=f"checkpoint-{checkpoint_step}",
                                trust_remote_code=True,
                                ignore_mismatched_sizes=True,
                                rope_scaling=None  # Disable RoPE scaling
                            )
                            print(f"Successfully loaded with {model_class.__name__}")
                            break
                        except Exception as e:
                            print(f"Failed with {model_class.__name__}: {e}")
                    else:
                        raise ValueError("All model loading attempts failed")
                    
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return {}

    # Ensure model has padding token config
    if hasattr(model, 'config') and tokenizer.pad_token is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Move model to device and set evaluation mode
    model.to(device)
    model.eval()
    if use_fp16 and device.type == 'cuda':
        model = model.half()

    results = {}

    # Using the global config module here, import it inside the function to be sure
    import config

    # Evaluate all tasks
    for task_name in config.DATASET_PATHS.keys():
        if task_name in model_name:
            print(f"Evaluating task: {task_name}")
            task_metrics = evaluate_model_on_task(model, tokenizer, task_name, device)
            
            if task_metrics:
                model_path = f"{model_name}-checkpoint-{checkpoint_step}" if checkpoint_step > 0 else origin_model
                results[task_name] = EvaluationResult(
                    model_name=model_path,
                    task=task_name,
                    rouge1=task_metrics["rouge1"],
                    rouge2=task_metrics["rouge2"],
                    rougeL=task_metrics["rougeL"],
                    bleu=task_metrics["bleu"],
                    exact_match=task_metrics["exact_match"],
                    f1=task_metrics["f1"],
                    accuracy=task_metrics["accuracy"]
                )
            print(f"model_name: {model_name}, checkpoint_step: {checkpoint_step}, task_name: {task_name}, results: {results}")

    # Release memory
    del model
    if 'finetuned_model' in locals():
        del finetuned_model
    if 'original_model' in locals():
        del original_model
    torch.cuda.empty_cache()

    return results

def main():
    # Set device and precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = torch.cuda.is_available()  # Enable half-precision if GPU is available
    
    # Define path to the base model
    origin_model = "meta-llama/Llama-3.2-1B"
    
    # Load tokenizer for preprocessing
    tokenizer = AutoTokenizer.from_pretrained(origin_model)
    if tokenizer.pad_token is None:
        # Set padding token if not defined (common for LLaMA models)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Models to evaluate (can be modified as needed)
    models = ["full_ft"]  # Currently only evaluating full fine-tuned model
    # Commented out additional models for future use
    # models = ["mnli_ft", "mnli_lora", "qqp_ft", "qqp_lora", 
    #         "similar_ft", "similar_lora_4", "similar_lora_8", 
    #         "similar_lora_16", "similar_lora_32", "similar_lora_64"]
    
    all_results = []  # Store all evaluation results
    
    # Evaluate all checkpoints for each model
    for model_name in models:
        print(f"\n===== Evaluating model: {model_name} =====")
        
        # Determine model type (LoRA or full fine-tuning)
        is_lora = 'lora' in model_name
        model_config = {
            "name": SIMILAR_TASK_CONFIGS[model_name]["name"],  # HuggingFace model path
            "type": SIMILAR_TASK_CONFIGS[model_name]["type"],  # Model adaptation type
            "task_type": SIMILAR_TASK_CONFIGS[model_name]["task_type"],  # Assuming classification task
            # Extract LoRA rank from model name if applicable
            "rank": int(model_name.split("_")[-1]) if is_lora and "_" in model_name and model_name.split("_")[-1].isdigit() else None
        }
        
        # Get available checkpoint steps
        checkpoint_steps = get_checkpoint_steps(SIMILAR_TASK_CONFIGS[model_name]["name"], model_config["type"])
        # Filter to only include checkpoints after step 9654
        checkpoint_steps = [step for step in checkpoint_steps if step >= 9654]
        
        # Evaluate each selected checkpoint
        for step in checkpoint_steps:
            checkpoint_results = evaluate_checkpoint(
                model_config,  # Configuration dictionary
                step,          # Checkpoint step
                origin_model,  # Base model path
                tokenizer,     # Tokenizer
                device,        # CPU or GPU
                use_fp16       # Whether to use half precision
            )
            
            # Add results from each evaluation task
            for task, result in checkpoint_results.items():
                all_results.append(result.to_dict())
    
    # Save results to JSON file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"evaluation_results_{timestamp}.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n===== Evaluation complete! Results saved to evaluation_results_{timestamp}.json =====")

if __name__ == "__main__":
    main()