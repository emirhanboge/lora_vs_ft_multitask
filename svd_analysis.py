"""
SVD Analysis of LoRA and Full Fine-tuning

This script performs Singular Value Decomposition (SVD) analysis on the weight matrices
of LoRA and fully fine-tuned models to study representational drift from the pretrained model.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from checkpoint_utils import (
    load_checkpoint_model,
    load_base_model,
    SIMILAR_TASK_CONFIGS,
    DISSIMILAR_TASK_CONFIGS,
    get_all_checkpoints,
    test_similar_task_model,
    test_dissimilar_task_model
)
import seaborn as sns
from tqdm import tqdm
import logging
import json
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_layer_weights(model: torch.nn.Module, layer_name: str) -> Optional[torch.Tensor]:
    """
    Extract weights from a specific layer of the model.
    
    Args:
        model: The PyTorch model
        layer_name: Name of the layer (q_proj or v_proj)
    
    Returns:
        Optional[torch.Tensor]: Weight matrix of the specified layer
    """
    # For LoRA models, we need to compute the effective weight matrix
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        # Get LoRA config from the PEFT model
        if hasattr(model, 'peft_config'):
            # Get the active adapter name and config
            adapter_name = list(model.peft_config.keys())[0]  # Get first adapter
            lora_config = model.peft_config[adapter_name]
            scaling = lora_config.lora_alpha / lora_config.r
        else:
            scaling = 1.0
            logger.warning("Could not find LoRA config, using default scaling of 1.0")
        
        # Find the target layer
        target_layer = None
        target_lora = None
        
        for name, module in model.named_modules():
            if layer_name in name:
                if hasattr(module, 'weight'):
                    target_layer = module
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    target_lora = module
                if target_layer is not None and target_lora is not None:
                    break
        
        if target_layer is None:
            logger.error(f"Could not find base layer for {layer_name}")
            return None
            
        # Get base weights
        base_weights = target_layer.weight.detach()
        logger.info(f"Base weights device for {layer_name}: {base_weights.device}")
        
        # If we have LoRA weights, compute the update
        if target_lora is not None:
            lora_a = target_lora.lora_A.default.weight.detach()  # [r, in_features]
            lora_b = target_lora.lora_B.default.weight.detach()  # [out_features, r]
            logger.info(f"LoRA weights devices for {layer_name}: A: {lora_a.device}, B: {lora_b.device}")
            
            # Compute LoRA update: (B × A) × scaling
            lora_delta = torch.matmul(lora_b, lora_a) * scaling
            
            # Verify shapes match
            if lora_delta.shape != base_weights.shape:
                logger.error(f"Shape mismatch: base weights {base_weights.shape}, LoRA delta {lora_delta.shape}")
                return None
                
            # Add to base weights
            return base_weights + lora_delta
        else:
            return base_weights
    
    # For base model or full fine-tuned model
    else:
        for name, module in model.named_modules():
            if layer_name in name and hasattr(module, 'weight'):
                weights = module.weight.detach()
                logger.info(f"Full fine-tuning weights device for {layer_name}: {weights.device}")
                return weights
        
        logger.error(f"No weights found for layer {layer_name}")
        return None

def compute_svd(weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute SVD of weight matrix.
    
    Args:
        weights: Weight matrix
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: U, Σ, V matrices
    """
    U, S, V = torch.linalg.svd(weights.float(), full_matrices=False)
    return U, S, V

def compute_representational_drift(U_ft: torch.Tensor, U_pretrained: torch.Tensor, top_k: int = 32) -> float:
    """
    Compute representational drift between fine-tuned and pretrained singular vectors,
    focusing on the top-k singular vectors which capture the most important directions.
    We use a fixed top-k value across all models to ensure fair comparison between
    full fine-tuning and LoRA models of different ranks.
    
    Args:
        U_ft: Left singular vectors of fine-tuned model
        U_pretrained: Left singular vectors of pretrained model
        top_k: Number of top singular vectors to consider (default: 32)
               Fixed across all models for consistent comparison
    
    Returns:
        float: Drift value between 0 and 1, where higher values indicate
              greater deviation from the pretrained model's representation
    """
    # Take only the top-k singular vectors
    U_ft_top = U_ft[:, :top_k]
    U_pre_top = U_pretrained[:, :top_k]
    
    # Compute Frobenius norm of the difference between top-k vectors
    diff_norm = torch.norm(U_ft_top - U_pre_top, p='fro')
    denominator = torch.norm(U_ft_top, p='fro') + torch.norm(U_pre_top, p='fro')
    
    return (diff_norm / denominator).item()

def analyze_model_drift(
    task_type: str,
    model_name: str,
    checkpoints: List[int],
    base_model_U: Dict[str, torch.Tensor],
    top_k: int
) -> Dict[str, List[float]]:
    """
    Analyze representational drift across training checkpoints.
    
    Args:
        task_type: Type of task (similar/dissimilar)
        model_name: Name of the model configuration
        checkpoints: List of checkpoint steps to analyze
        base_model_U: Dictionary of base model's singular vectors for each layer
        top_k: Number of top singular vectors to consider
    
    Returns:
        Dict[str, List[float]]: Drift values for each layer across checkpoints
    """
    drift_values = {'q_proj': [], 'v_proj': []}
    
    logger.info(f"Using top-{top_k} singular vectors for drift analysis")
    
    for checkpoint in tqdm(checkpoints, desc=f"Analyzing {model_name}"):
        try:
            model = load_checkpoint_model(task_type, model_name, checkpoint)
            if model is None:
                logger.error(f"Failed to load model {model_name} at checkpoint {checkpoint}")
                for layer_name in ['q_proj', 'v_proj']:
                    drift_values[layer_name].append(float('nan'))
                continue
            
            for layer_name in ['q_proj', 'v_proj']:
                try:
                    weights = get_layer_weights(model, layer_name)
                    if weights is None:
                        logger.error(f"Failed to get weights for {layer_name} at checkpoint {checkpoint}")
                        drift_values[layer_name].append(float('nan'))
                        continue
                    
                    logger.info(f"Computing SVD for {layer_name} at checkpoint {checkpoint}")
                    U, _, _ = compute_svd(weights)
                    drift = compute_representational_drift(U, base_model_U[layer_name], top_k=top_k)
                    drift_values[layer_name].append(drift)
                    logger.info(f"Drift for {layer_name} at checkpoint {checkpoint} (top-{top_k}): {drift:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error processing {layer_name} at checkpoint {checkpoint}: {str(e)}")
                    drift_values[layer_name].append(float('nan'))
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error processing checkpoint {checkpoint}: {str(e)}")
            for layer_name in ['q_proj', 'v_proj']:
                drift_values[layer_name].append(float('nan'))
    
    return drift_values

def plot_drift_analysis(
    drift_results: Dict[str, Dict[str, Dict[str, List[float]]]], 
    checkpoints: List[int],
    output_path: str,
    top_k: int
):
    """
    Plot drift analysis results in two subplots without smoothing.
    Shows how different models deviate from the pretrained model's singular vectors.
    
    Args:
        drift_results: Nested dictionary containing drift values
        checkpoints: List of checkpoint steps
        output_path: Path to save the plot
        top_k: Number of top singular vectors used for this analysis
    """
    # Set up seaborn style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    # Set up colors for different configurations
    colors = sns.color_palette("husl", n_colors=len(drift_results))
    
    # Create x-axis intervals
    x_intervals = list(range(1, 11))  # 1 to 10
    
    # Determine y-axis limits from data
    all_drifts = []
    for model_results in drift_results.values():
        all_drifts.extend([d for d in model_results['q_proj'] if not np.isnan(d)])
        all_drifts.extend([d for d in model_results['v_proj'] if not np.isnan(d)])
    y_min = min(all_drifts) * 0.95
    y_max = max(all_drifts) * 1.05
    
    # Plot q_proj in first subplot
    for i, (model_name, layers) in enumerate(drift_results.items()):
        q_proj_drifts = layers['q_proj']
        # Format label to show rank or full fine-tuning
        if 'lora' in model_name:
            rank = int(model_name.split('_')[1])
            label = f"LoRA (r={rank})"
        else:
            label = "Full Fine-tuning"
        ax1.plot(x_intervals, q_proj_drifts, label=label, color=colors[i], 
                marker='o', markersize=8, linewidth=2)
    
    ax1.set_ylabel('Representational Drift', fontsize=12)
    ax1.set_title('Q Projection Layer', fontsize=14, pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.set_ylim(y_min, y_max)
    ax1.tick_params(axis='both', labelsize=10)
    
    # Plot v_proj in second subplot
    for i, (model_name, layers) in enumerate(drift_results.items()):
        v_proj_drifts = layers['v_proj']
        # Format label to show rank or full fine-tuning
        if 'lora' in model_name:
            rank = int(model_name.split('_')[1])
            label = f"LoRA (r={rank})"
        else:
            label = "Full Fine-tuning"
        ax2.plot(x_intervals, v_proj_drifts, label=label, color=colors[i], 
                marker='s', markersize=8, linewidth=2)
    
    ax2.set_xlabel('Training Interval', fontsize=12)
    ax2.set_ylabel('Representational Drift', fontsize=12)
    ax2.set_title('V Projection Layer', fontsize=14, pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax2.set_ylim(y_min, y_max)
    ax2.tick_params(axis='both', labelsize=10)
    
    # Set integer ticks for x-axis
    ax2.set_xticks(x_intervals)
    
    # Add overall title
    fig.suptitle(f'Representational Drift Analysis (Top-{top_k})', fontsize=16, y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def save_results_to_json(
    similar_results: Dict[str, Dict[str, List[float]]],
    dissimilar_results: Dict[str, Dict[str, List[float]]],
    similar_checkpoints: Dict[str, List[int]],
    dissimilar_checkpoints: Dict[str, List[int]],
    output_path: str = "drift_analysis_results.json"
):
    """
    Save drift analysis results to a JSON file.
    
    Args:
        similar_results: Results for similar tasks
        dissimilar_results: Results for dissimilar tasks
        similar_checkpoints: Checkpoints for similar tasks
        dissimilar_checkpoints: Checkpoints for dissimilar tasks
        output_path: Path to save the JSON file
    """
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "description": "Drift analysis results comparing LoRA and full fine-tuning"
        },
        "similar_tasks": {
            "results": {},
            "checkpoints": {k: list(map(int, v)) for k, v in similar_checkpoints.items()},
            "checkpoint_indices": {}
        },
        "dissimilar_tasks": {
            "results": {},
            "checkpoints": {k: list(map(int, v)) for k, v in dissimilar_checkpoints.items()},
            "checkpoint_indices": {}
        }
    }
    
    # Process similar tasks results
    for model_name, model_results in similar_results.items():
        results["similar_tasks"]["results"][model_name] = model_results
        # Add checkpoint indices (0-based) for plotting
        if model_name in similar_checkpoints:
            results["similar_tasks"]["checkpoint_indices"][model_name] = list(range(len(similar_checkpoints[model_name])))
    
    # Process dissimilar tasks results
    for model_name, model_results in dissimilar_results.items():
        results["dissimilar_tasks"]["results"][model_name] = model_results
        # Add checkpoint indices (0-based) for plotting
        if model_name in dissimilar_checkpoints:
            results["dissimilar_tasks"]["checkpoint_indices"][model_name] = list(range(len(dissimilar_checkpoints[model_name])))
    
    # Convert all numpy/torch values to Python native types
    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, (float, int)):
            return obj
        elif obj is None:
            return None
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)
    
    results = convert_to_native(results)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

def standardize_checkpoints(checkpoints: Dict[str, List[int]]) -> Dict[str, List[int]]:
    """
    Standardize checkpoints across models by taking the first 10 checkpoints.
    
    Args:
        checkpoints: Dictionary mapping model names to their checkpoint steps
    
    Returns:
        Dict[str, List[int]]: Dictionary with standardized checkpoint steps
    """
    standardized = {}
    
    for model_name, steps in checkpoints.items():
        if not steps:
            continue
            
        # Take first 10 checkpoints
        standardized_steps = steps[:10]
        standardized[model_name] = standardized_steps
        logger.info(f"Standardized checkpoints for {model_name}: {standardized_steps}")
    
    return standardized

def main():
    # Create output directory for plots
    os.makedirs("drift_analysis_plots", exist_ok=True)
    
    # Load base model and compute its singular vectors
    logger.info("Loading base model and computing singular vectors...")
    base_model = load_base_model("similar", device_map="cuda")  # Task type doesn't matter for base model
    base_model_U = {}
    for layer_name in ['q_proj', 'v_proj']:
        weights = get_layer_weights(base_model, layer_name)
        U, _, _ = compute_svd(weights)
        base_model_U[layer_name] = U
    
    # Clean up base model
    del base_model
    torch.cuda.empty_cache()
    
    # Get available checkpoints for each task type
    similar_checkpoints = get_all_checkpoints("similar")
    dissimilar_checkpoints = get_all_checkpoints("dissimilar")
    
    # Standardize checkpoints
    similar_checkpoints = standardize_checkpoints(similar_checkpoints)
    dissimilar_checkpoints = standardize_checkpoints(dissimilar_checkpoints)
    
    # Different top-k values to analyze
    top_k_values = [4, 8, 16, 32, 64, 128, 256]
    
    # Analyze and plot for each top-k value
    for top_k in top_k_values:
        logger.info(f"\nAnalyzing with top-{top_k} singular vectors")
        
        # Analyze similar tasks
        similar_results = {}
        """
        for model_name in ['lora_4', 'lora_8', 'lora_16', 'lora_32', 'lora_64', 'full_ft']:
            if model_name in similar_checkpoints:
                logger.info(f"\nAnalyzing similar task model {model_name}")
                drift_values = analyze_model_drift('similar', model_name, similar_checkpoints[model_name], base_model_U, top_k)
                similar_results[model_name] = drift_values
        
        # Plot results for similar tasks
        if similar_results:
            plot_drift_analysis(
                similar_results, 
                similar_checkpoints[list(similar_results.keys())[0]],
                f'drift_analysis_plots/similar_tasks_top{top_k}.png',
                top_k
            )
        """
        
        # Analyze dissimilar tasks
        dissimilar_results = {}
        for model_name in ['lora_4', 'lora_8', 'lora_16', 'lora_32', 'lora_64', 'full_ft']:
            if model_name in dissimilar_checkpoints:
                logger.info(f"\nAnalyzing dissimilar task model {model_name}")
                drift_values = analyze_model_drift('dissimilar', model_name, dissimilar_checkpoints[model_name], base_model_U, top_k)
                dissimilar_results[model_name] = drift_values
        
        # Plot results for dissimilar tasks
        if dissimilar_results:
            plot_drift_analysis(
                dissimilar_results, 
                dissimilar_checkpoints[list(dissimilar_results.keys())[0]],
                f'drift_analysis_plots/dissimilar_tasks_top{top_k}.png',
                top_k
            )
        
        # Save results to JSON for this top-k
        save_results_to_json(
            similar_results,
            dissimilar_results,
            similar_checkpoints,
            dissimilar_checkpoints,
            f"drift_analysis_plots/drift_analysis_top{top_k}.json"
        )

if __name__ == "__main__":
    main() 