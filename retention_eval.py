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
import config
import os
import traceback
from checkpoint_utils import load_model_for_checkpoint, get_checkpoint_steps

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
# Set random seeds for reproducibility
seed=42
set_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Enable hardware acceleration optimizations
# TF32 is a precision format that's faster than FP32 but more accurate than FP16
torch.backends.cuda.matmul.allow_tf32 = True  # For GPUs like A100 that support TF32
torch.backends.cudnn.benchmark = True         # Optimize cudnn for fixed-size inputs

# Configuration dictionary for different model variants trained on dissimilar tasks (QA and code summarization)
DISSIMILAR_TASK_CONFIGS = {
    "full_ft": {
        "name": "emirhanboge/LLaMA_1B_qa_code_summarization_dissimilar_FullFT",
        "type": "full_ft",  # Full fine-tuning
        "task_type": "dissimilar"
    },
    "lora_4": {
        "name": "emirhanboge/LLaMA_1B_qa_code_summarization_dissimilar_LoRA_4",
        "type": "lora",  # LoRA adapter with rank 4
        "rank": 4,
        "task_type": "dissimilar"
    },
    "lora_8": {
        "name": "emirhanboge/LLaMA_1B_qa_code_summarization_dissimilar_LoRA_8",
        "type": "lora",  # LoRA adapter with rank 8
        "rank": 8,
        "task_type": "dissimilar"
    },
    "lora_16": {
        "name": "emirhanboge/LLaMA_1B_qa_code_summarization_dissimilar_LoRA_16",
        "type": "lora",  # LoRA adapter with rank 16
        "rank": 16,
        "task_type": "dissimilar"
    },
    "lora_32": {
        "name": "emirhanboge/LLaMA_1B_qa_code_summarization_dissimilar_LoRA_32",
        "type": "lora",  # LoRA adapter with rank 32
        "rank": 32,
        "task_type": "dissimilar"
    },
    "lora_64": {
        "name": "emirhanboge/LLaMA_1B_qa_code_summarization_dissimilar_LoRA_64",
        "type": "lora",  # LoRA adapter with rank 64
        "rank": 64,
        "task_type": "dissimilar"
    }
}

# Paths to benchmark datasets for evaluating model knowledge retention
RETENTION_DATASET_PATHS = {
    "hellaswag": "Rowan/hellaswag",  # Multiple-choice commonsense reasoning dataset
    "arc_challenge": "ibragim-bad/arc_challenge",  # Multiple-choice science questions
    "winogrande": "liyucheng/winogrande_val"  # Commonsense reasoning with pronoun resolution
}

# Custom collate function to efficiently batch dataset examples
def dynamic_collate_fn(batch):
    # Convert batch data to tensors efficiently (numpy first for better performance)
    input_ids = torch.tensor(np.array([item["input_ids"] for item in batch]), dtype=torch.long)
    attention_mask = torch.tensor(np.array([item["attention_mask"] for item in batch]), dtype=torch.long)
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    question_ids = torch.tensor([item["question_ids"] for item in batch], dtype=torch.long)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "question_ids": question_ids
    }

def evaluate_all_dataset(model, tokenizer, batch_size=64):
    """
    Evaluates model performance on all benchmark datasets.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for text preprocessing
        batch_size: Number of examples per batch
        
    Returns:
        Dictionary containing evaluation results for each dataset and overall accuracy
    """
    model.eval()  # Set model to evaluation mode
    device = model.device
    dataset_results = {}

    total_correct = 0
    total_samples = 0
    
    # Move model to GPU if available and enable data parallelism for multi-GPU setup
    if torch.cuda.is_available():
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    
    # Evaluate each dataset separately
    for dataset_name in RETENTION_DATASET_PATHS:
        print(f"\nEvaluating {dataset_name}...")
        
        # Load dataset from Hugging Face hub
        dataset = load_dataset(RETENTION_DATASET_PATHS[dataset_name], split="validation")

        # Define dataset-specific preprocessing function
        def preprocess_function(examples):
            processed = {"input_ids": [], "attention_mask": [], "labels": [], "question_ids": []}
            
            # Process HellaSwag dataset format
            if dataset_name == "hellaswag":
                num_choices = 4
                for q_id, (ctx, endings, label) in enumerate(zip(examples["ctx"], examples["endings"], examples["label"])):
                    for opt_idx, ending in enumerate(endings):
                        text = f"{ctx} {ending}"
                        encoded = tokenizer(
                            text,
                            max_length=256,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt"
                        )
                        # Store tokenized inputs as numpy arrays for efficiency
                        processed["input_ids"].append(encoded["input_ids"].squeeze(0).numpy())
                        processed["attention_mask"].append(encoded["attention_mask"].squeeze(0).numpy())
                        processed["labels"].append(label)
                        processed["question_ids"].append(q_id)
            
            # Process ARC Challenge dataset format
            elif dataset_name == "arc_challenge":
                num_choices = 4
                for q_id, (question, choices, answer) in enumerate(zip(examples["question"], [choice["text"] for choice in examples["choices"]], examples["answerKey"])):
                    label = ord(answer) - ord("A")  # Convert letter answer to index (A=0, B=1, etc.)
                    for opt_idx, choice in enumerate(choices):
                        text = f"Question: {question}\nAnswer: {choice}"
                        encoded = tokenizer(
                            text,
                            max_length=256,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt"
                        )
                        processed["input_ids"].append(encoded["input_ids"].squeeze(0).numpy())
                        processed["attention_mask"].append(encoded["attention_mask"].squeeze(0).numpy())
                        processed["labels"].append(label)
                        processed["question_ids"].append(q_id)
            
            # Process Winogrande dataset format
            elif dataset_name == "winogrande":
                num_choices = 2
                for q_id, (sentence, op1, op2, ans) in enumerate(zip(examples["sentence"], examples["option1"], examples["option2"], examples["answer"])):
                    label = int(ans) - 1  # Convert 1/2 answer to 0/1 index
                    for opt_idx, option in enumerate([op1, op2]):
                        text = sentence.replace("_", option)  # Replace blank with option
                        encoded = tokenizer(
                            text,
                            max_length=256,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt"
                        )
                        processed["input_ids"].append(encoded["input_ids"].squeeze(0).numpy())
                        processed["attention_mask"].append(encoded["attention_mask"].squeeze(0).numpy())
                        processed["labels"].append(label)
                        processed["question_ids"].append(q_id)
            
            return processed
        
        # Apply preprocessing to dataset
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=256,
            num_proc=4,  # Parallel processing
            remove_columns=dataset.column_names,
            load_from_cache_file=False,  # Don't use cached preprocessed dataset
            features=datasets.Features({
                "input_ids": datasets.Sequence(datasets.Value("int32")),
                "attention_mask": datasets.Sequence(datasets.Value("int32")),
                "labels": datasets.Value("int64"),
                "question_ids": datasets.Value("int64")
            })
        )
        
        # Set number of choices based on the dataset
        if dataset_name == "hellaswag":
            num_choices = 4
        elif dataset_name == "arc_challenge":
            num_choices = 4
        elif dataset_name == "winogrande":
            num_choices = 2
        else:
            num_choices = 4  # Default fallback

        # Create dataloader for batched processing
        dataloader = DataLoader(
            processed_dataset.with_format("numpy"),
            batch_size=batch_size,
            collate_fn=dynamic_collate_fn,
            pin_memory=True,  # Speed up data transfer to GPU
            num_workers=0  # Single process data loading
        )
        
        # Evaluation metrics
        correct = 0
        total = 0
        question_predictions = {}
        
        # Import for detailed debugging if needed
        import traceback
        
        # Disable gradient calculation for evaluation to save memory and compute
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
                # Move batch to device (CPU/GPU)
                batch = {k: v.to(device) for k, v in batch.items()}
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                question_ids = batch["question_ids"]
                
                try:
                    # Run model with automatic mixed precision for faster inference on GPU
                    with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    # Debug model outputs if needed (only for first question)
                    if question_ids[0].item() == 0:
                        # Debug code commented out but available for troubleshooting
                        # Print model output attributes if they are tensors
                        attrs = [attr for attr in dir(outputs) if not attr.startswith('_')]
                        for attr in attrs:
                            try:
                                value = getattr(outputs, attr)
                                if isinstance(value, torch.Tensor):
                                    print(f"Attribute {attr} shape: {value.shape}")
                            except:
                                pass
                    
                    # Get logits from model output
                    logits = outputs.logits
                    
                    # Handle different output formats based on logits dimensions
                    if logits.dim() == 3:  # 3D: [batch_size, sequence_length, vocab_size]
                        # Language model approach: Calculate perplexity-based scores
                        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
                        shift_labels = input_ids[:, 1:].reshape(-1)
                        
                        # Calculate cross-entropy loss for each token position
                        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                        token_losses = loss_fct(shift_logits, shift_labels)
                        
                        # Calculate sequence-level loss (average over valid tokens)
                        token_losses = token_losses.reshape(input_ids.size(0), -1)
                        seq_lengths = attention_mask.sum(dim=1) - 1
                        seq_lengths = torch.clamp(seq_lengths, min=1)  # Avoid division by zero
                        valid_tokens = attention_mask[:, 1:].float()
                        masked_losses = token_losses * valid_tokens
                        seq_loss = masked_losses.sum(dim=1) / seq_lengths
                        
                        # Convert loss to score (negative loss = higher score is better)
                        scores = -seq_loss.detach().cpu()
                        
                    elif logits.dim() == 2:  # 2D: [batch_size, some_dimension]
                        # Handle two cases: classifier output or token logits
                        
                        # Case 1: Large output dimension indicates vocabulary distribution
                        if logits.size(1) > 1000:  
                            # Use maximum logit value as score
                            scores = logits.max(dim=1).values.detach().cpu()
                        
                        # Case 2: Small output dimension indicates classification task
                        else:
                            # Use maximum class logit as score
                            scores = logits.max(dim=1).values.detach().cpu()
                    
                    # Case 3: Direct loss output from model
                    elif hasattr(outputs, 'loss') and outputs.loss is not None:
                        # Use negative loss as score
                        scores = -outputs.loss.detach().cpu().unsqueeze(0)  # Ensure 1D tensor
                        
                    else:  # Fallback case
                        # If unexpected output format, use zeros as scores
                        scores = torch.zeros(input_ids.size(0))
                        
                except Exception as e:
                    # Error handling with detailed trace
                    print(f"Error processing model output: {e}")
                    traceback.print_exc()
                    print("Using fallback scoring method")
                    scores = torch.zeros(input_ids.size(0))
                
                # Group scores by question ID to handle multiple-choice evaluation
                batch_size = input_ids.size(0)
                for i in range(batch_size):
                    q_id = question_ids[i].item()
                    score_val = scores[i].item() if scores.numel() > i else 0.0
                    label = labels[i].item()
                    
                    # Initialize question entry if not exists
                    if q_id not in question_predictions:
                        question_predictions[q_id] = {
                            "scores": [None] * num_choices,
                            "label": label
                        }
                    # Calculate current choice index and store score
                    choice_idx = len([s for s in question_predictions[q_id]["scores"] if s is not None]) % num_choices
                    question_predictions[q_id]["scores"][choice_idx] = score_val
        
        # Calculate accuracy by selecting highest-scoring option for each question
        for q_id, pred_info in question_predictions.items():
            scores = pred_info["scores"]
            label = pred_info["label"]
            if None not in scores:  # Only process complete sets of choices
                prediction = np.argmax(scores)
                if prediction == label:
                    correct += 1
                total += 1
        
        # Calculate accuracy for this dataset
        accuracy = correct / total if total > 0 else 0
        dataset_results[dataset_name] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
        
        # Add to running totals for overall accuracy
        total_correct += correct
        total_samples += total
        print(f"data: {dataset_name}, accuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Calculate overall accuracy across all datasets
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    # Return complete evaluation results
    return {
        "dataset_results": dataset_results,
        "overall_accuracy": overall_accuracy
    }

def evaluate_all_checkpoints(origin_path, model_name, tokenizer):
    """
    Evaluates all available checkpoints for a model.
    
    Args:
        origin_path: Path to the original base model
        model_name: Name of the fine-tuned model
        tokenizer: Tokenizer for text preprocessing
        
    Returns:
        List of evaluation results for each checkpoint
    """
    results = []
    # Get available checkpoint steps
    try:
        # Get checkpoint steps from utility function
        checkpoint_steps = get_checkpoint_steps(model_name, "lora" if "LoRA" in model_name else "full_ft")
        
        if not checkpoint_steps:
            print(f"No checkpoints found for {model_name}, evaluating main model only")
            checkpoint_steps = ["main"]
        else:
            print(f"Found checkpoints for {model_name}: {checkpoint_steps}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_fp16 = True
        is_lora = 'LoRA' in model_name
        
        for step in checkpoint_steps:
            try:
                if step == "main":
                    print(f"\n===== Evaluating {model_name} main model =====")
                    
                    # Load main model (no specific checkpoint)
                    if is_lora:
                        # For LoRA models, load base model first then apply adapter
                        print("Loading as LoRA model from main branch")
                        original_model = AutoModelForCausalLM.from_pretrained(origin_path)
                        
                        # Ensure padding token is set
                        if tokenizer.pad_token is None:
                            tokenizer.pad_token = tokenizer.eos_token
                            original_model.config.pad_token_id = tokenizer.pad_token_id
                        
                        # Apply LoRA adapter and merge weights
                        finetuned_model = PeftModel.from_pretrained(
                            original_model,
                            model_name,
                            trust_remote_code=True
                        )
                        reverted_model = finetuned_model.merge_and_unload()
                    else:
                        # For full fine-tuned models, load directly
                        print("Loading as full fine-tuned model from main branch")
                        
                        try:
                            # Try loading as causal language model first
                            reverted_model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                trust_remote_code=True,
                                ignore_mismatched_sizes=True,
                                rope_scaling=None  # Disable RoPE scaling to avoid issues
                            )
                        except Exception as e:
                            # Fall back to sequence classification model
                            print(f"Failed to load as CausalLM: {e}")
                            reverted_model = AutoModelForSequenceClassification.from_pretrained(
                                model_name,
                                trust_remote_code=True,
                                ignore_mismatched_sizes=True
                            )
                else:
                    print(f"\n===== Evaluating {model_name} at step {step} =====")
                    
                    # First try to download and inspect the config file for this checkpoint
                    try:
                        from huggingface_hub import hf_hub_download
                        import json
                        import os
                        
                        try:
                            print(f"Attempting to access config for checkpoint at {model_name}/checkpoint-{step}")
                            # Get the config file for this checkpoint
                            config_path = hf_hub_download(
                                repo_id=model_name,
                                filename="config.json",
                                subfolder=f"checkpoint-{step}"
                            )
                            
                            print(f"Successfully found config at: {config_path}")
                            
                            # Read the config to determine model architecture
                            with open(config_path, 'r') as f:
                                model_config_json = json.load(f)
                            
                            # Get model type info from config
                            model_architecture = model_config_json.get("architectures", ["LlamaForCausalLM"])[0]
                            print(f"Model architecture from config: {model_architecture}")
                            
                            # Fix the RoPE scaling configuration issue that can cause problems
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
                            checkpoint_dir = os.path.join(cache_dir, f"checkpoint-{step}")
                            print(f"Using local cache path: {checkpoint_dir}")
                            
                            # Load the model based on config architecture
                            if is_lora:
                                # For LoRA models
                                print("Loading as LoRA model")
                                original_model = AutoModelForCausalLM.from_pretrained(origin_path)
                                
                                # Ensure padding token is set
                                if tokenizer.pad_token is None:
                                    tokenizer.pad_token = tokenizer.eos_token
                                    original_model.config.pad_token_id = tokenizer.pad_token_id
                                
                                # Load checkpoint-specific LoRA adapter
                                finetuned_model = PeftModel.from_pretrained(
                                    original_model, 
                                    checkpoint_dir,
                                    trust_remote_code=True,
                                    device_map="auto" if torch.cuda.is_available() else None
                                )
                                reverted_model = finetuned_model.merge_and_unload()
                            else:
                                # For full fine-tuned models, load based on architecture
                                if "CausalLM" in model_architecture:
                                    print("Loading as Causal LM model")
                                    reverted_model = AutoModelForCausalLM.from_pretrained(
                                        checkpoint_dir,
                                        trust_remote_code=True,
                                        ignore_mismatched_sizes=True,
                                        device_map="auto" if torch.cuda.is_available() else None,
                                        rope_scaling=None  # Disable RoPE scaling
                                    )
                                elif "SequenceClassification" in model_architecture:
                                    print("Loading as Sequence Classification model")
                                    reverted_model = AutoModelForSequenceClassification.from_pretrained(
                                        checkpoint_dir,
                                        trust_remote_code=True,
                                        ignore_mismatched_sizes=True,
                                        device_map="auto" if torch.cuda.is_available() else None,
                                        rope_scaling=None  # Disable RoPE scaling
                                    )
                                else:
                                    # Default to causal LM if architecture is unclear
                                    print(f"Unknown architecture: {model_architecture}, trying Causal LM")
                                    reverted_model = AutoModelForCausalLM.from_pretrained(
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
                            if is_lora:
                                # Alternative LoRA loading method
                                print("Trying alternative LoRA loading")
                                original_model = AutoModelForCausalLM.from_pretrained(
                                    origin_path,
                                    rope_scaling=None  # Disable RoPE scaling
                                )
                                
                                # Ensure padding token
                                if tokenizer.pad_token is None:
                                    tokenizer.pad_token = tokenizer.eos_token
                                    original_model.config.pad_token_id = tokenizer.pad_token_id
                                
                                # Try with repo_id and revision format
                                finetuned_model = PeftModel.from_pretrained(
                                    original_model,
                                    model_name,
                                    subfolder=f"checkpoint-{step}",
                                    trust_remote_code=True
                                )
                                reverted_model = finetuned_model.merge_and_unload()
                            else:
                                # Try direct loading with modified config
                                try:
                                    print("Trying to load with modified config")
                                    
                                    # Download config first
                                    from transformers import AutoConfig
                                    try:
                                        model_auto_config = AutoConfig.from_pretrained(
                                            f"{model_name}/checkpoint-{step}",
                                            trust_remote_code=True
                                        )
                                        # Remove or fix RoPE scaling
                                        if hasattr(model_auto_config, "rope_scaling"):
                                            delattr(model_auto_config, "rope_scaling")
                                        
                                        # Now load model with the modified config
                                        reverted_model = AutoModelForCausalLM.from_pretrained(
                                            f"{model_name}/checkpoint-{step}",
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
                                            reverted_model = model_class.from_pretrained(
                                                model_name,
                                                subfolder=f"checkpoint-{step}",
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
                        continue  # Skip this checkpoint if loading fails
                
                # Ensure model knows padding token
                if hasattr(reverted_model, 'config'):
                    reverted_model.config.pad_token_id = tokenizer.pad_token_id
                
                # Move model to device and set evaluation mode
                reverted_model.to(device)
                reverted_model.eval()
                if use_fp16 and device.type == 'cuda':
                    reverted_model = reverted_model.half()  # Use half precision for faster inference
                
                # Evaluate the model
                checkpoint_result = evaluate_all_dataset(reverted_model, tokenizer)
                results.append({
                    "model": model_name,
                    "step": step,
                    "model_type": "LoRA" if is_lora else "FullFT",  
                    "results": checkpoint_result
                })
                
            except Exception as e:
                print(f"Error evaluating {model_name} at step {step}: {e}")
                import traceback
                print(traceback.format_exc())
            
            # Release memory to avoid CUDA out-of-memory errors
            if 'reverted_model' in locals():
                del reverted_model
            if 'finetuned_model' in locals():
                del finetuned_model
            if 'original_model' in locals():
                del original_model
            torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"Failed to evaluate checkpoints for model {model_name}: {e}")
        import traceback
        print(traceback.format_exc())
    
    return results

# Usage in main function:
def main():
    # Use mixed precision calculation
    use_fp16 = True
    
    origin_model = "meta-llama/Llama-3.2-1B"
    
    for model_name in DISSIMILAR_TASK_CONFIGS.keys():
        print(f"\n===== Evaluating model: {model_name} =====")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(origin_model)
        
        # Ensure tokenizer has padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Evaluate main model instead of trying to access checkpoints
        results = evaluate_all_checkpoints(origin_model, DISSIMILAR_TASK_CONFIGS[model_name]["name"], tokenizer)

        print("\n===== Evaluation Results =====")
        # Rest of code remains the same

        print("\n===== Evaluation Results =====")
        for result in results:
            model_identifier = result["model"]
            step = result["step"]
            model_type = result.get("model_type", "Original")
            dataset_results = result["results"]["dataset_results"]
            overall_accuracy = result["results"]["overall_accuracy"]
            
            if model_identifier == "original":
                print(f"Original model:")
            else:
                print(f"Model {model_identifier} ({model_type}) at step {step}:")
                
            for dataset_name, dataset_result in dataset_results.items():
                print(f"  {dataset_name}: accuracy = {dataset_result['accuracy']:.4f} "
                      f"({dataset_result['correct']}/{dataset_result['total']})")
            
            print(f"  Overall accuracy: {overall_accuracy:.4f}\n")
        
        # Release GPU memory
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()