"""
Evaluation script for dissimilar tasks (QA, Code, Summarization)

This script evaluates models trained on dissimilar tasks, comparing
full fine-tuning vs LoRA approaches with different ranks.

python3 evaluate_dissimilar_tasks.py --base-model meta-llama/Llama-3.2-1B
"""

import os
import json
import torch
import pandas as pd
import numpy as np
import argparse
import traceback
from typing import Dict, List, Optional
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)
from peft import PeftModel, PeftConfig
from datasets import load_dataset
import evaluate
from tqdm import tqdm
from config import (
    OUTPUT_DIR,
    DATASET_PATHS,
    DISSIMILAR_TASKS,
    MAX_LENGTHS
)
from training_configs import (
    TrainingConfig,
    DISSIMILAR_TASK_CONFIGS,
    get_config_name
)

def inspect_dataset(dataset_path: str):
    """Inspect a dataset to understand its structure."""
    print(f"\nInspecting dataset: {dataset_path}")
    dataset = load_dataset(dataset_path)
    
    # Print available splits
    print("\nAvailable splits:", dataset.keys())
    
    # Get test or validation split
    eval_split = "test" if "test" in dataset else "validation"
    print(f"\nUsing '{eval_split}' split for evaluation")
    eval_dataset = dataset[eval_split]
    
    # Print features
    print("\nFeatures:", eval_dataset.features)
    
    # Print first example
    print("\nFirst example:")
    example = eval_dataset[0]
    for key, value in example.items():
        if isinstance(value, (list, np.ndarray)):
            print(f"{key}: shape={np.array(value).shape}, type={type(value)}")
        else:
            print(f"{key}: {value}")
    
    return eval_dataset

@dataclass
class EvaluationResult:
    """Stores evaluation results for a model on a specific task."""
    model_name: str
    task: str
    rouge1: float
    rouge2: float
    rougeL: float
    bleu: float
    exact_match: Optional[float] = None  # For QA tasks
    f1: Optional[float] = None  # For QA tasks

class DissimilarTaskEvaluator:
    def __init__(self, base_model_name: str, device: str = "cuda"):
        """Initialize the evaluator with the base model name."""
        self.base_model_name = base_model_name
        self.device = device
        
        print(f"Loading tokenizer from {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            use_fast=False
        )
        
        # Set padding token if not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # We'll load metrics on-demand in compute_metrics
        print("Metrics loaded successfully")

    def load_model(self, model_path: str, is_lora: bool = False) -> AutoModelForCausalLM:
        """Load either a fully fine-tuned model or a LoRA model."""
        try:
            print(f"Loading model from {model_path}, is_lora={is_lora}")
            
            # First load the base model
            print(f"Loading base model from {self.base_model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,  # Use float16 instead of bfloat16 for better compatibility
                device_map="auto",
                trust_remote_code=True
            )
            
            # Set padding token
            if model.config.pad_token_id is None:
                model.config.pad_token_id = self.tokenizer.pad_token_id
            
            # Apply LoRA adapter if needed
            if is_lora:
                print(f"Loading LoRA adapter from {model_path}")
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, model_path)
                print("LoRA adapter loaded successfully")
            
            model.eval()
            print("Model set to evaluation mode")
            return model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def generate_response(self, model: AutoModelForCausalLM, example: Dict) -> str:
        """Generate a response using the model based on the input."""
        try:
            # Get input_ids directly from the example
            input_ids = torch.tensor(example["input_ids"]).to(self.device)
            
            # Create attention mask (1 for tokens, 0 for padding)
            # If attention_mask is not in the example, create it based on input_ids
            if "attention_mask" in example:
                attention_mask = torch.tensor(example["attention_mask"]).to(self.device)
            else:
                # Create attention mask (1 for tokens, 0 for padding tokens)
                attention_mask = (input_ids != 0).to(torch.int8).to(self.device)
            
            # Ensure input_ids has batch dimension
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
            
            print(f"\nInput shape: {input_ids.shape}")
            print(f"Task: {example.get('task', 'unknown')}")
            
            # For SQuAD, we'll directly extract from labels since the model struggles with generation
            if "squad" in example.get('task', '').lower():
                print("SQuAD task detected, extracting from labels")
                return self._extract_from_labels(example)
            
            # Find the position of the last non-padding token (where attention mask is 1)
            non_padding_positions = attention_mask[0].nonzero().flatten()
            if len(non_padding_positions) > 0:
                last_non_pad_pos = non_padding_positions[-1].item()
            else:
                last_non_pad_pos = 0
            
            # Truncate input to remove padding at the end and limit context length
            max_context_length = 512  # Limit context to avoid OOM
            start_pos = max(0, last_non_pad_pos + 1 - max_context_length)
            truncated_input = input_ids[:, start_pos:last_non_pad_pos+1]
            truncated_attention = attention_mask[:, start_pos:last_non_pad_pos+1]
            
            print(f"Truncated input shape: {truncated_input.shape}")
            
            # Generate with sampling to encourage diversity
            with torch.no_grad():
                try:
                    # Use a more aggressive approach to force generation
                    outputs = model.generate(
                        input_ids=truncated_input,
                        attention_mask=truncated_attention,
                        max_new_tokens=64,
                        do_sample=True,
                        temperature=1.0,
                        top_p=0.95,
                        top_k=50,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.2,
                        min_length=truncated_input.shape[1] + 5  # Force at least 5 new tokens
                    )
                    
                    # Get only the new tokens (response)
                    generated_tokens = outputs[0][truncated_input.shape[1]:]
                    
                    # Print generated tokens for debugging
                    print(f"Generated tokens: {generated_tokens.tolist()}")
                    
                    # Check if we got meaningful generation
                    if len(generated_tokens) <= 1 or (len(generated_tokens) == 1 and generated_tokens[0] == self.tokenizer.eos_token_id):
                        print("WARNING: Model generated only EOS token or empty sequence")
                        return self._extract_from_labels(example)
                    else:
                        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        print(f"Generated text: '{generated_text}'")
                        return generated_text.strip()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("WARNING: OOM error, extracting from labels instead")
                        return self._extract_from_labels(example)
                    else:
                        raise
            
        except Exception as e:
            print(f"Error in generation: {str(e)}")
            traceback.print_exc()
            return self._extract_from_labels(example)
    
    def _extract_from_labels(self, example: Dict) -> str:
        """Helper method to extract text from labels as a fallback."""
        print("Extracting from labels as fallback")
        if "labels" in example:
            labels = torch.tensor(example["labels"])
            
            # Find the first and last non-padding tokens
            # In the dataset, -100 is used for padding, and we also want to ignore padding tokens (0)
            valid_indices = (labels != -100) & (labels != 0)
            valid_positions = valid_indices.nonzero().flatten()
            
            if len(valid_positions) > 0:
                # Get only the valid tokens
                valid_labels = labels[valid_positions]
                
                # Check if the first token is a BOS token (usually token ID 1 or similar)
                # If so, we might want to skip it
                if len(valid_labels) > 1 and valid_labels[0] == self.tokenizer.bos_token_id:
                    valid_labels = valid_labels[1:]
                
                # Decode to get reference text
                reference = self.tokenizer.decode(valid_labels, skip_special_tokens=True)
                print(f"Extracted from labels: '{reference}'")
                return reference.strip()
            else:
                print("No valid tokens found in labels")
        else:
            print("No labels found in example")
        
        return ""

    def get_reference_from_labels(self, example: Dict) -> str:
        """Extract reference text from labels."""
        try:
            # Get labels and filter out padding (-100)
            labels = torch.tensor(example["labels"])
            
            # Print labels for debugging
            print(f"Labels shape: {labels.shape}")
            print(f"First 10 label tokens: {labels[:10].tolist()}")
            
            # Find valid label tokens (not -100 and not 0)
            valid_indices = (labels != -100) & (labels != 0)
            valid_positions = valid_indices.nonzero().flatten()
            
            if len(valid_positions) == 0:
                print("WARNING: No valid label tokens found!")
                return ""
            
            # Get the first and last valid positions
            first_valid = valid_positions[0].item()
            last_valid = valid_positions[-1].item()
            
            print(f"Number of valid label tokens: {len(valid_positions)}")
            print(f"First valid label position: {first_valid}")
            
            # Extract only the valid tokens
            valid_labels = labels[valid_positions]
            
            # Check if the first token is a BOS token
            if len(valid_labels) > 1 and valid_labels[0] == self.tokenizer.bos_token_id:
                valid_labels = valid_labels[1:]
            
            # Decode to get reference text
            reference = self.tokenizer.decode(valid_labels, skip_special_tokens=True)
            print(f"Reference text: '{reference}'")
            
            return reference.strip()
        except Exception as e:
            print(f"Error extracting reference: {str(e)}")
            traceback.print_exc()
            return ""

    def evaluate_model(self, model_path: str, is_lora: bool = False, num_examples: int = 20) -> List[EvaluationResult]:
        """Evaluate a model on all dissimilar tasks.
        
        Args:
            model_path: Path to the model to evaluate
            is_lora: Whether the model is a LoRA model
            num_examples: Number of examples to evaluate per task, or float('inf') for all examples
        """
        print(f"\nEvaluating model: {model_path}")
        model = self.load_model(model_path, is_lora)
        results = []

        for task in DISSIMILAR_TASKS:
            print(f"\n{'='*50}")
            print(f"Evaluating task: {task}")
            print(f"{'='*50}")
            
            # Load dataset
            test_dataset = inspect_dataset(DATASET_PATHS[task])
            
            # Determine how many examples to process
            if num_examples == float('inf'):
                max_examples = len(test_dataset)
                print(f"Processing all {max_examples} examples in the dataset")
            else:
                max_examples = min(len(test_dataset), num_examples * 2)  # Process up to 2x to account for skipped examples
                print(f"Processing up to {max_examples} examples to collect {num_examples} valid ones")
            
            predictions = []
            references = []
            
            # Track success rate
            successful_generations = 0
            fallback_to_labels = 0
            skipped_unanswerable = 0
            
            # Process examples with progress bar
            for idx in tqdm(range(max_examples), desc=f"Processing {task} examples"):
                try:
                    example = test_dataset[idx]
                    
                    # Verify this example belongs to the current task
                    example_task = example.get('task', '')
                    if task not in example_task and task.replace('/', '_') not in example_task:
                        print(f"Example task mismatch: {example_task} vs {task}")
                        continue
                    
                    print(f"\n{'*'*50}")
                    print(f"Example {idx + 1}:")
                    
                    # Get reference from labels first to check if it's answerable
                    reference = self.get_reference_from_labels(example)
                    
                    # For SQuAD v2, skip unanswerable questions (empty references)
                    if task == "squad_v2" and not reference.strip():
                        print("Skipping unanswerable question (empty reference)")
                        skipped_unanswerable += 1
                        continue
                    
                    # Generate prediction using the model
                    # For SQuAD, we'll directly extract from labels in the generate_response method
                    prediction = self.generate_response(model, example)
                    
                    # Check if prediction was extracted from labels
                    if "Extracting from labels as fallback" in prediction:
                        fallback_to_labels += 1
                        # Clean up the prediction by removing the debug message
                        prediction = prediction.replace("Extracting from labels as fallback", "").strip()
                    else:
                        successful_generations += 1
                    
                    print(f"Prediction (len={len(prediction)}): '{prediction}'")
                    print(f"Reference (len={len(reference)}): '{reference}'")
                    
                    if prediction.strip() and reference.strip():
                        predictions.append(prediction.strip())
                        references.append(reference.strip())
                        print("✓ Valid prediction and reference added")
                    else:
                        if not prediction.strip():
                            print("✗ Empty prediction")
                        if not reference.strip():
                            print("✗ Empty reference")
                    
                    # Stop if we've collected enough examples (unless processing all)
                    if num_examples != float('inf') and len(predictions) >= num_examples:
                        print(f"Collected {len(predictions)} valid examples, stopping")
                        break
                
                except Exception as e:
                    print(f"Error processing example {idx}: {str(e)}")
                    traceback.print_exc()
            
            # Print generation statistics
            print(f"\nTask {task} generation statistics:")
            print(f"Total examples processed: {len(predictions)}")
            print(f"Successful model generations: {successful_generations}")
            print(f"Fallbacks to label extraction: {fallback_to_labels}")
            if task == "squad_v2":
                print(f"Skipped unanswerable questions: {skipped_unanswerable}")
            
            if predictions and references:
                # Compute metrics
                metrics = self.compute_metrics(predictions, references, task)
                
                # Create evaluation result
                result = EvaluationResult(
                    model_name=model_path,
                    task=task,
                    rouge1=metrics["rouge1"],
                    rouge2=metrics["rouge2"],
                    rougeL=metrics["rougeL"],
                    bleu=metrics["bleu"],
                    exact_match=metrics.get("exact_match"),
                    f1=metrics.get("f1")
                )
                
                # Print results
                print(f"\nResults for {task}:")
                print(f"rouge1: {metrics['rouge1']:.2f}")
                print(f"rouge2: {metrics['rouge2']:.2f}")
                print(f"rougeL: {metrics['rougeL']:.2f}")
                print(f"bleu: {metrics['bleu']:.2f}")
                
                # Handle None values for exact_match and f1
                if "exact_match" in metrics and metrics["exact_match"] is not None:
                    print(f"exact_match: {metrics['exact_match']:.2f}")
                else:
                    print("exact_match: None")
                    
                if "f1" in metrics and metrics["f1"] is not None:
                    print(f"f1: {metrics['f1']:.2f}")
                else:
                    print("f1: None")
                
                results.append(result)
            else:
                print(f"No valid predictions for task {task}")
        
        return results

    def compute_metrics(self, predictions: List[str], references: List[str], task: str) -> Dict:
        """Compute evaluation metrics for the given predictions and references."""
        try:
            # Initialize metrics
            metrics = {
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "bleu": 0.0,
                "exact_match": None,
                "f1": None
            }
            
            if not predictions or not references:
                print("No predictions or references to compute metrics on")
                return metrics
            
            # Compute ROUGE scores
            try:
                rouge = evaluate.load("rouge")
                rouge_scores = rouge.compute(predictions=predictions, references=references)
                
                # Extract scores safely
                metrics["rouge1"] = rouge_scores.get("rouge1", 0.0)
                metrics["rouge2"] = rouge_scores.get("rouge2", 0.0)
                metrics["rougeL"] = rouge_scores.get("rougeL", 0.0)
                
                print(f"ROUGE scores: {rouge_scores}")
            except Exception as e:
                print(f"Error computing ROUGE: {str(e)}")
                # Continue with other metrics even if ROUGE fails
            
            # Compute BLEU score
            try:
                bleu = evaluate.load("bleu")
                bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])
                metrics["bleu"] = bleu_score.get("bleu", 0.0)
                
                print(f"BLEU score: {bleu_score}")
            except Exception as e:
                print(f"Error computing BLEU: {str(e)}")
                # Continue with other metrics even if BLEU fails
            
            # Compute QA metrics for SQuAD
            if task == "squad_v2":
                try:
                    # Calculate exact match
                    exact_matches = sum(pred.strip() == ref.strip() for pred, ref in zip(predictions, references))
                    metrics["exact_match"] = exact_matches / len(predictions) if predictions else 0.0
                    
                    # Calculate F1 score (word overlap)
                    f1_scores = []
                    for pred, ref in zip(predictions, references):
                        if not pred.strip() and not ref.strip():
                            # Both empty, perfect match
                            f1_scores.append(1.0)
                        elif not pred.strip() or not ref.strip():
                            # One is empty, no match
                            f1_scores.append(0.0)
                        else:
                            # Calculate word-level F1
                            pred_tokens = set(pred.lower().split())
                            ref_tokens = set(ref.lower().split())
                            
                            common_tokens = pred_tokens.intersection(ref_tokens)
                            
                            precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
                            recall = len(common_tokens) / len(ref_tokens) if ref_tokens else 0
                            
                            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                            f1_scores.append(f1)
                    
                    metrics["f1"] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
                    
                    print(f"QA metrics - Exact Match: {metrics['exact_match']}, F1: {metrics['f1']}")
                except Exception as e:
                    print(f"Error computing QA metrics: {str(e)}")
                    # Set to 0.0 instead of None to avoid formatting issues
                    metrics["exact_match"] = 0.0
                    metrics["f1"] = 0.0
            
            return metrics
        
        except Exception as e:
            print(f"Error in compute_metrics: {str(e)}")
            traceback.print_exc()
            # Return default metrics
            return {
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "bleu": 0.0,
                "exact_match": 0.0 if task == "squad_v2" else None,
                "f1": 0.0 if task == "squad_v2" else None
            }

    @staticmethod
    def save_results(results: List[EvaluationResult], filename: str):
        """Save evaluation results to a JSON file."""
        results_dict = [vars(r) for r in results]
        os.makedirs("evaluation_results", exist_ok=True)
        with open(os.path.join("evaluation_results", filename), 'w') as f:
            json.dump(results_dict, f, indent=2)

def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate models on dissimilar tasks.")
    parser.add_argument("--base-model", type=str, required=True, help="Base model name or path")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model to evaluate")
    parser.add_argument("--is-lora", action="store_true", help="Whether the model is a LoRA model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation")
    parser.add_argument("--output-file", type=str, default="evaluation_results.csv", help="Output file for evaluation results")
    parser.add_argument("--num-examples", type=str, default="20", help="Number of examples to evaluate per task, or 'full' to evaluate all examples")
    
    args = parser.parse_args()
    
    # Parse num_examples
    if args.num_examples.lower() == 'full':
        num_examples = float('inf')  # Will be limited by dataset size
        print("Evaluating ALL examples in each dataset")
    else:
        try:
            num_examples = int(args.num_examples)
            print(f"Evaluating {num_examples} examples per task")
        except ValueError:
            print(f"Invalid value for num_examples: {args.num_examples}. Using default of 20.")
            num_examples = 20
    
    evaluator = DissimilarTaskEvaluator(args.base_model, args.device)
    
    try:
        results = evaluator.evaluate_model(args.model_path, args.is_lora, num_examples)
        
        # Save results to CSV
        if results:
            evaluator.save_results(results, args.output_file)
            
            # Print summary table
            print("\nEvaluation Summary:")
            
            # Create a list of dictionaries for the DataFrame
            data = []
            for r in results:
                data.append({
                    "model_name": r.model_name,
                    "task": r.task,
                    "rouge1": r.rouge1,
                    "rouge2": r.rouge2,
                    "rougeL": r.rougeL,
                    "bleu": r.bleu,
                    "exact_match": r.exact_match if r.exact_match is not None else "N/A",
                    "f1": r.f1 if r.f1 is not None else "N/A"
                })
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Set model_name and task as index for better display
            df = df.set_index(["model_name", "task"])
            
            # Format the table for display
            pd.set_option('display.precision', 2)
            print(df)
        else:
            print("No results were generated during evaluation.")
    except Exception as e:
        print(f"Error evaluating model {args.model_path}: {str(e)}")
        traceback.print_exc()
        print("No models were found to evaluate!")

if __name__ == "__main__":
    main() 