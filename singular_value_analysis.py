"""
Singular Value Analysis Script

This script analyzes the singular values and vectors of model weights across different
checkpoints, comparing LoRA and full fine-tuning approaches with the pre-trained model
for both similar (classification) and dissimilar (generation) tasks.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import list_repo_files
import torch.distributed._shard.checkpoint as dist_cp
from torch.distributed._shard.checkpoint import FileSystemReader as dist_cpFileSystemReader

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Model configurations
SIMILAR_TASK_CONFIGS = {
    "full_ft": {
        "name": "emirhanboge/LLaMA_1B_sst2_mnli_qqp_FullFT",
        "num_checkpoints": 3,
        "type": "full_ft",
        "task_type": "similar"
    },
    "lora_4": {
        "name": "emirhanboge/LLaMA_1B_sst2_mnli_qqp_LoRA_4",
        "num_checkpoints": 10,
        "type": "lora",
        "rank": 4,
        "task_type": "similar"
    },
    "lora_8": {
        "name": "emirhanboge/LLaMA_1B_sst2_mnli_qqp_LoRA_8",
        "num_checkpoints": 10,
        "type": "lora",
        "rank": 8,
        "task_type": "similar"
    },
    "lora_16": {
        "name": "emirhanboge/LLaMA_1B_sst2_mnli_qqp_LoRA_16",
        "num_checkpoints": 10,
        "type": "lora",
        "rank": 16,
        "task_type": "similar"
    },
    "lora_32": {
        "name": "emirhanboge/LLaMA_1B_sst2_mnli_qqp_LoRA_32",
        "num_checkpoints": 10,
        "type": "lora",
        "rank": 32,
        "task_type": "similar"
    },
    "lora_64": {
        "name": "emirhanboge/LLaMA_1B_sst2_mnli_qqp_LoRA_64",
        "num_checkpoints": 10,
        "type": "lora",
        "rank": 64,
        "task_type": "similar"
    }
}

DISSIMILAR_TASK_CONFIGS = {
    "full_ft": {
        "name": "emirhanboge/LLaMA_1B_qa_code_sum_FullFT_",
        "num_checkpoints": 3,
        "type": "full_ft",
        "task_type": "dissimilar"
    },
    "lora_4": {
        "name": "emirhanboge/LLaMA_1B_qa_code_sum_LoRA_4",
        "num_checkpoints": 10,
        "type": "lora",
        "rank": 4,
        "task_type": "dissimilar"
    },
    "lora_8": {
        "name": "emirhanboge/LLaMA_1B_qa_code_sum_LoRA_8",
        "num_checkpoints": 10,
        "type": "lora",
        "rank": 8,
        "task_type": "dissimilar"
    },
    "lora_16": {
        "name": "emirhanboge/LLaMA_1B_qa_code_sum_LoRA_16",
        "num_checkpoints": 10,
        "type": "lora",
        "rank": 16,
        "task_type": "dissimilar"
    },
    "lora_32": {
        "name": "emirhanboge/LLaMA_1B_qa_code_sum_LoRA_32",
        "num_checkpoints": 10,
        "type": "lora",
        "rank": 32,
        "task_type": "dissimilar"
    },
    "lora_64": {
        "name": "emirhanboge/LLaMA_1B_qa_code_sum_LoRA_64",
        "num_checkpoints": 10,
        "type": "lora",
        "rank": 64,
        "task_type": "dissimilar"
    }
}

def get_sample_inputs(task_type):
    """Get sample inputs for model computation."""
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")
    
    try:
        if task_type == "similar":
            # Classification example (sentiment analysis)
            inputs = tokenizer(
                "This movie is great!",
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
        else:
            # Generation example (QA task)
            inputs = tokenizer(
                "Question: What is the capital of France? Answer:",
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
        
        # Move inputs to device
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        return inputs
    
    except Exception as e:
        print(f"Error in tokenization: {e}")
        raise

def get_weight_matrices(model):
    """Extract weight matrices from model for SVD analysis."""
    try:
        weight_matrices = {}
        
        # Convert named_modules to list first to get progress bar
        print("Extracting weight matrices...")
        modules = list(model.named_modules())
        
        # Only analyze q_proj and v_proj where LoRA was applied
        target_layers = ['q_proj', 'v_proj']
        
        # For attention layers where LoRA was applied
        for name, module in tqdm(modules, desc="Processing layers"):
            if any(layer in name.lower() for layer in target_layers):
                if hasattr(module, 'weight'):
                    # Move to CPU and convert to numpy
                    weight = module.weight.detach().cpu().float().numpy()
                    weight_matrices[name] = weight
        
        if not weight_matrices:
            print("Warning: No weight matrices found in model")
        else:
            print("Successfully extracted weight matrices")
        return weight_matrices
    
    except Exception as e:
        print(f"Error extracting weight matrices: {e}")
        raise

def compute_svd_metrics(weight_matrix, pretrained_weight=None):
    """Compute SVD metrics for a weight matrix."""
    try:
        U, S, Vh = np.linalg.svd(weight_matrix, full_matrices=False)
        
        metrics = {
            "singular_values": S,
            "left_vectors": U,
            "right_vectors": Vh,
            "effective_rank": np.sum(S > 1e-5),  # Number of significant singular values
            "spectral_norm": float(S[0]),  # Convert to float for serialization
            "condition_number": float(S[0] / S[-1]) if S[-1] > 1e-10 else float('inf'),
            "rank_95_energy": int(np.argmax(np.cumsum(S) / np.sum(S) >= 0.95) + 1)  # Rank needed for 95% energy
        }
        
        if pretrained_weight is not None:
            # Compute SVD for pretrained weights
            U_pre, S_pre, Vh_pre = np.linalg.svd(pretrained_weight, full_matrices=False)
            
            # Compare singular value distributions
            metrics.update({
                "sv_correlation": float(spearmanr(S[:min(len(S), len(S_pre))], 
                                          S_pre[:min(len(S), len(S_pre))])[0]),
                "left_space_similarity": float(np.mean([
                    np.abs(cosine_similarity(U[:, i:i+1], U_pre[:, i:i+1]))[0][0]
                    for i in range(min(U.shape[1], U_pre.shape[1]))
                ])),
                "right_space_similarity": float(np.mean([
                    np.abs(cosine_similarity(Vh[i:i+1], Vh_pre[i:i+1]))[0][0]
                    for i in range(min(Vh.shape[0], Vh_pre.shape[0]))
                ]))
            })
            
            # Principal angles between subspaces
            k = min(10, U.shape[1], U_pre.shape[1])  # Use top-k singular vectors
            Q1 = U[:, :k]
            Q2 = U_pre[:, :k]
            principal_angles = np.arccos(np.clip(np.linalg.svd(Q1.T @ Q2)[1], -1.0, 1.0))
            metrics["mean_principal_angle"] = float(np.mean(principal_angles))
        
        return metrics
    
    except Exception as e:
        print(f"Error computing SVD metrics: {e}")
        raise

def get_checkpoint_steps(repo_name):
    """Get available checkpoint steps from HuggingFace repo."""
    try:
        print(f"\nInspecting repository: {repo_name}")
        files = list_repo_files(repo_name)
        
        steps = []
        
        # For full fine-tuning models, look for model.safetensors files
        if repo_name.endswith("FullFT"):
            for file in files:
                if "model.safetensors" in file:
                    try:
                        step = int(file.split("/")[0].replace("checkpoint-", ""))
                        if step not in steps:
                            steps.append(step)
                    except ValueError:
                        continue
        else:
            # For LoRA models, look for adapter_config.json in checkpoint folders
            for file in files:
                if "checkpoint-" in file and "adapter_config.json" in file:
                    try:
                        step = int(file.split("checkpoint-")[1].split("/")[0])
                        if step not in steps:
                            steps.append(step)
                    except ValueError:
                        continue
        
        if steps:
            print(f"\nFound checkpoint steps: {sorted(steps)}")
        else:
            print("\nWarning: No valid checkpoints found in repository")
            print("Looking for:")
            if repo_name.endswith("FullFT_"):
                print("  - Full fine-tuning: checkpoint-X/model.safetensors")
            else:
                print("  - LoRA: checkpoint-X/adapter_config.json")
        
        return sorted(steps)
    
    except Exception as e:
        print(f"Error inspecting repository {repo_name}: {e}")
        return []

def load_model(config, step=None):
    """Load model based on configuration."""
    try:
        print(f"\nLoading model for config: {config}")
        print(f"Step: {step}")
        
        if config["task_type"] == "similar":
            model_class = AutoModelForSequenceClassification
            model_kwargs = {"num_labels": 3}  # MNLI has 3 labels
        else:
            model_class = AutoModelForCausalLM
            model_kwargs = {}
        
        if step is None:
            # Load pretrained model
            print("Loading base pretrained model...")
            model = model_class.from_pretrained(
                "meta-llama/Llama-3.2-1B",
                torch_dtype=torch.float16,
                device_map="auto",  # Let the model decide optimal device mapping
                **model_kwargs
            )
        elif config["type"] == "lora":
            # Load LoRA model
            print("Loading LoRA model...")
            # First load base model
            base_model = model_class.from_pretrained(
                "meta-llama/Llama-3.2-1B",
                torch_dtype=torch.float16,
                device_map="auto",  # Let the model decide optimal device mapping
                **model_kwargs
            )
            
            # For LoRA models, use the root adapter_config.json
            adapter_path = config['name']
            print(f"Loading adapter from root: {adapter_path}")
            
            try:
                # Load PEFT config from root
                peft_config = PeftConfig.from_pretrained(adapter_path)
                print(f"Loaded PEFT config: {peft_config}")
                
                # Load PEFT model with adapter
                model = PeftModel.from_pretrained(
                    base_model,
                    adapter_path,
                    is_trainable=False
                )
                print("Successfully loaded PEFT model")
                
                # Merge adapter weights with base model for analysis
                print("Merging adapter weights...")
                model = model.merge_and_unload()
                
            except Exception as e:
                print(f"Error loading adapter: {e}")
                raise
            
        else:
            # Load full fine-tuned model from main branch with checkpoint path
            print("Loading full fine-tuned model...")
            
            try:
                # Load from the main branch with checkpoint subfolder
                repo_id = config['name']
                subfolder = f"checkpoint-{step}"
                print(f"Loading from repo: {repo_id}, subfolder: {subfolder}")
                
                model = model_class.from_pretrained(
                    repo_id,
                    subfolder=subfolder,
                    revision="main",
                    torch_dtype=torch.float16,
                    device_map="auto",  # Let the model decide optimal device mapping
                    trust_remote_code=True,
                    **model_kwargs
                )
                print("Successfully loaded checkpoint")
                
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                raise
        
        # Ensure model is in eval mode
        model.eval()
        print("Model loaded and ready for inference")
        return model
    
    except Exception as e:
        print(f"Error in load_model: {e}")
        raise

def analyze_singular_values(task_type="both"):
    """Analyze singular values across models and checkpoints."""
    try:
        results = []
        output_dir = "svd_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine which configurations to analyze
        if task_type == "similar":
            configs = SIMILAR_TASK_CONFIGS
        elif task_type == "dissimilar":
            configs = DISSIMILAR_TASK_CONFIGS
        else:  # both
            configs = {**SIMILAR_TASK_CONFIGS, **DISSIMILAR_TASK_CONFIGS}
        
        # Initialize results with at least one row to prevent DataFrame errors
        results.append({
            "model": "initialization",
            "checkpoint": 0,
            "layer": "none",
            "type": "none",
            "task_type": task_type,
            "rank": 0,
            "effective_rank": 0,
            "rank_95_energy": 0,
            "spectral_norm": 0,
            "condition_number": 0,
            "sv_correlation": 0,
            "left_space_similarity": 0,
            "right_space_similarity": 0,
            "mean_principal_angle": 0,
            **{f"sv_{i}": 0 for i in range(10)}
        })
        
        # Load pretrained models
        print("\nLoading pretrained models...")
        pretrained_similar = load_model({"task_type": "similar"}) if task_type == "similar" else None
        pretrained_dissimilar = load_model({"task_type": "dissimilar"}) if task_type == "dissimilar" else None
        pretrained_weights = {
            "similar": get_weight_matrices(pretrained_similar) if pretrained_similar else None,
            "dissimilar": get_weight_matrices(pretrained_dissimilar) if pretrained_dissimilar else None
        }

        print(f"Starting analysis for {task_type} tasks...")
        for model_name, config in tqdm(configs.items(), desc="Analyzing models"):
            try:
                # Get available checkpoint steps
                checkpoint_steps = get_checkpoint_steps(config["name"])
                if not checkpoint_steps:
                    print(f"No checkpoints found for {model_name}, skipping...")
                    continue
                
                for step in tqdm(checkpoint_steps, desc=f"Analyzing {model_name} checkpoints"):
                    try:
                        # Load checkpoint
                        model = load_model(config, step)
                        
                        # Get weight matrices
                        weight_matrices = get_weight_matrices(model)
                        
                        # Get corresponding pretrained weights
                        pretrained = pretrained_weights[config["task_type"]]
                        
                        # Analyze each weight matrix
                        for name, weight in weight_matrices.items():
                            pretrained_weight = pretrained.get(name)
                            metrics = compute_svd_metrics(weight, pretrained_weight)
                            
                            # Store results
                            result = {
                                "model": model_name,
                                "checkpoint": step,
                                "layer": name,
                                "type": config["type"],
                                "task_type": config["task_type"],
                                "rank": config.get("rank"),
                                **{f"sv_{i}": float(sv) for i, sv in enumerate(metrics["singular_values"][:10])},
                                "effective_rank": metrics["effective_rank"],
                                "rank_95_energy": metrics["rank_95_energy"],
                                "spectral_norm": metrics["spectral_norm"],
                                "condition_number": metrics["condition_number"]
                            }
                            
                            if pretrained_weight is not None:
                                result.update({
                                    "sv_correlation": metrics["sv_correlation"],
                                    "left_space_similarity": metrics["left_space_similarity"],
                                    "right_space_similarity": metrics["right_space_similarity"],
                                    "mean_principal_angle": metrics["mean_principal_angle"]
                                })
                            
                            results.append(result)
                        
                        # Clear GPU memory
                        del model
                        torch.cuda.empty_cache()
                    
                    except Exception as e:
                        print(f"Error processing checkpoint {step} for {model_name}: {e}")
                        continue
            
            except Exception as e:
                print(f"Error processing model {model_name}: {e}")
                continue
        
        # Create DataFrame and remove initialization row if we have actual results
        results_df = pd.DataFrame(results)
        if len(results_df) > 1:
            results_df = results_df[results_df["model"] != "initialization"]
        
        if len(results_df) == 0:
            raise ValueError("No results were collected. Check the model configurations and checkpoints.")
        
        return results_df, output_dir
    
    except Exception as e:
        print(f"Error in analyze_singular_values: {e}")
        raise

def create_visualizations(results_df, output_dir):
    """Create visualizations for singular value analysis."""
    try:
        # Set style
        plt.style.use('default')
        
        # Create visualizations for each task type
        for task_type in results_df["task_type"].unique():
            task_df = results_df[results_df["task_type"] == task_type]
            task_dir = os.path.join(output_dir, task_type)
            os.makedirs(task_dir, exist_ok=True)
            
            print(f"\nCreating visualizations for {task_type} tasks...")
            
            # 1. Singular value decay plots
            plt.figure(figsize=(15, 8))
            for model in task_df["model"].unique():
                data = task_df[task_df["model"] == model].iloc[0]
                singular_values = [data[f"sv_{i}"] for i in range(10)]
                plt.plot(range(1, 11), singular_values, label=model, marker='o')
            plt.yscale('log')
            plt.xlabel("Singular Value Index")
            plt.ylabel("Singular Value (log scale)")
            plt.title(f"Singular Value Decay Across Models ({task_type} tasks)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(task_dir, "singular_value_decay.png"))
            plt.close()
            
            # 2. Effective rank comparison
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=task_df, x="model", y="effective_rank", hue="layer")
            plt.xticks(rotation=45)
            plt.xlabel("Model Configuration")
            plt.ylabel("Effective Rank")
            plt.title(f"Effective Rank Distribution ({task_type} tasks)")
            plt.tight_layout()
            plt.savefig(os.path.join(task_dir, "effective_rank_comparison.png"))
            plt.close()
            
            # 3. Similarity with pretrained model
            fig, axes = plt.subplots(2, 1, figsize=(12, 12))
            
            sns.boxplot(data=task_df, x="model", y="left_space_similarity", hue="layer", ax=axes[0])
            axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
            axes[0].set_title(f"Left Singular Vector Space Similarity ({task_type} tasks)")
            
            sns.boxplot(data=task_df, x="model", y="right_space_similarity", hue="layer", ax=axes[1])
            axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
            axes[1].set_title(f"Right Singular Vector Space Similarity ({task_type} tasks)")
            
            plt.tight_layout()
            plt.savefig(os.path.join(task_dir, "pretrained_similarity.png"))
            plt.close()
            
            # 4. Training dynamics
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            metrics = ["effective_rank", "rank_95_energy", "mean_principal_angle", "sv_correlation"]
            titles = ["Effective Rank", "95% Energy Rank", "Mean Principal Angle", "SV Correlation"]
            
            for ax, metric, title in zip(axes.flat, metrics, titles):
                for model in task_df["model"].unique():
                    data = task_df[task_df["model"] == model]
                    ax.plot(data["checkpoint"], data[metric], label=model, marker='o')
                ax.set_xlabel("Training Steps")
                ax.set_ylabel(title)
                ax.set_title(f"{title} Evolution ({task_type} tasks)")
                ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(task_dir, "training_dynamics.png"))
            plt.close()
            
            # 5. LoRA rank analysis
            lora_df = task_df[task_df["type"] == "lora"].copy()
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            metrics = ["effective_rank", "rank_95_energy", "spectral_norm", "condition_number"]
            titles = ["Effective Rank", "95% Energy Rank", "Spectral Norm", "Condition Number"]
            
            for ax, metric, title in zip(axes.flat, metrics, titles):
                sns.scatterplot(data=lora_df, x="rank", y=metric, hue="layer", ax=ax)
                ax.set_xlabel("LoRA Rank")
                ax.set_ylabel(title)
                ax.set_title(f"LoRA Rank vs {title} ({task_type} tasks)")
            
            plt.tight_layout()
            plt.savefig(os.path.join(task_dir, "lora_rank_analysis.png"))
            plt.close()
    
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        raise

def main():
    """Main function to run the analysis."""
    try:
        import argparse
        parser = argparse.ArgumentParser(description="Run singular value analysis")
        parser.add_argument(
            "--task-type",
            type=str,
            choices=["similar", "dissimilar", "both"],
            default="both",
            help="Type of tasks to analyze"
        )
        args = parser.parse_args()
        
        print(f"Starting singular value analysis for {args.task_type} tasks...")
        results_df, output_dir = analyze_singular_values(args.task_type)
        
        # Save raw results
        results_path = os.path.join(output_dir, "svd_analysis_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\nSaved raw results to {results_path}")
        
        print("\nCreating visualizations...")
        create_visualizations(results_df, output_dir)
        
        print(f"\nAnalysis complete! Check the output files in the {output_dir} directory.")
    
    except Exception as e:
        print(f"\nError in main: {e}")
        raise

if __name__ == "__main__":
    main() 