"""
Singular Vector Similarity Analysis Script

This script analyzes the similarity between singular vectors of LoRA and full fine-tuning
models compared to the pretrained model, tracking how this similarity evolves during training.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from transformers import LlamaForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import list_repo_files
import traceback
import argparse

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Model configurations
SIMILAR_TASK_CONFIGS = {
    "full_ft": {
        "name": "emirhanboge/LLaMA_1B_sst2_mnli_qqp_FullFT",
        "type": "full_ft",
        "task_type": "similar"
    },
    "lora_4": {
        "name": "emirhanboge/LLaMA_1B_sst2_mnli_qqp_LoRA_4",
        "type": "lora",
        "rank": 4,
        "task_type": "similar"
    },
    "lora_8": {
        "name": "emirhanboge/LLaMA_1B_sst2_mnli_qqp_LoRA_8",
        "type": "lora",
        "rank": 8,
        "task_type": "similar"
    },
    "lora_16": {
        "name": "emirhanboge/LLaMA_1B_sst2_mnli_qqp_LoRA_16",
        "type": "lora",
        "rank": 16,
        "task_type": "similar"
    },
    "lora_32": {
        "name": "emirhanboge/LLaMA_1B_sst2_mnli_qqp_LoRA_32",
        "type": "lora",
        "rank": 32,
        "task_type": "similar"
    },
    "lora_64": {
        "name": "emirhanboge/LLaMA_1B_sst2_mnli_qqp_LoRA_64",
        "type": "lora",
        "rank": 64,
        "task_type": "similar"
    }
}

def get_weight_matrices(model):
    """
    Extract weight matrices from the model for analysis.
    
    Args:
        model: The model to extract weights from
    
    Returns:
        Dictionary of weight matrices
    """
    try:
        weight_matrices = {}
        
        # Check if this is a LoRA model
        is_lora_model = hasattr(model, 'base_weights') and any('lora_A' in key for key in model.base_weights.keys() if isinstance(key, str))
        
        # For LoRA models, extract base weights and compute LoRA updates
        if is_lora_model:
            print("Detected LoRA model, extracting base weights and LoRA updates")
            
            # Get base weights
            base_weights = {}
            for name, weights in model.base_weights.items():
                if 'lora' not in name and any(layer in name for layer in ['q_proj', 'v_proj']):
                    base_weights[name] = weights
            
            # Extract LoRA A and B matrices
            lora_a_matrices = {}
            lora_b_matrices = {}
            
            for name, weights in model.base_weights.items():
                if 'lora_A' in name and any(layer in name for layer in ['q_proj', 'v_proj']):
                    lora_a_matrices[name] = weights
                elif 'lora_B' in name and any(layer in name for layer in ['q_proj', 'v_proj']):
                    lora_b_matrices[name] = weights
            
            # Compute LoRA updates (B*A) and store with corresponding base weights
            for a_name, a_matrix in lora_a_matrices.items():
                layer_name = a_name.split('.lora_A')[0]
                b_name = f"{layer_name}.lora_B"
                
                if b_name in lora_b_matrices:
                    b_matrix = lora_b_matrices[b_name]
                    base_name = f"{layer_name}.weight"
                    
                    if base_name in base_weights:
                        # Compute LoRA update: B*A
                        lora_update = np.matmul(b_matrix, a_matrix)
                        
                        # Store the update and base weight
                        weight_matrices[layer_name] = (lora_update, base_weights[base_name])
            
            print(f"Extracted {len(weight_matrices)} LoRA updates")
            return weight_matrices
        
        # For regular models, extract weights directly
        else:
            print("Extracting weights from regular model")
            
            for name, param in model.named_parameters():
                if any(layer in name for layer in ['q_proj', 'v_proj']) and 'weight' in name:
                    # Convert to numpy for analysis
                    weight = param.detach().cpu().float().numpy()
                    
                    # Use the layer name as the key
                    layer_name = name.split('.weight')[0]
                    weight_matrices[layer_name] = weight
            
            print(f"Extracted {len(weight_matrices)} weight matrices")
            return weight_matrices
    
    except Exception as e:
        print(f"Error extracting weight matrices: {e}")
        traceback.print_exc()
        return {}

def compute_weight_analysis(weight_matrix, base_weight, is_lora_update=False, lora_rank=None):
    """
    Comprehensive analysis of weight changes or LoRA updates.
    
    Args:
        weight_matrix: Weight matrix (for full FT) or LoRA update matrix BA (for LoRA)
        base_weight: Original/base weight matrix
        is_lora_update: Whether this is a LoRA update matrix
        lora_rank: Configured rank for LoRA (if applicable)
    """
    try:
        # Compute SVD of base weights
        U_base, S_base, Vh_base = np.linalg.svd(base_weight, full_matrices=False)
        
        if is_lora_update:
            # For LoRA, analyze the update matrix directly
            lora_update = weight_matrix
            
            # Compute SVD of the LoRA update
            U_lora, S_lora, Vh_lora = np.linalg.svd(lora_update, full_matrices=False)
            
            # Compute relative scale of update
            update_norm = np.linalg.norm(lora_update)
            base_norm = np.linalg.norm(base_weight)
            relative_magnitude = update_norm / base_norm
            
            # Analyze rank properties
            sv_threshold = 1e-5 * S_base[0]  # Threshold relative to largest base singular value
            effective_rank = np.sum(S_lora > sv_threshold)
            
            # Energy analysis
            total_energy = np.sum(S_lora**2)
            energy_95 = np.searchsorted(np.cumsum(S_lora**2) / total_energy, 0.95) + 1
            
            # For LoRA, we analyze the top-k singular vectors where k is the LoRA rank
            num_vectors = min(lora_rank, len(S_lora))
            
            # Project LoRA update onto base model's singular vectors
            # This tells us how much the update aligns with the original weight space
            projection_matrix = np.zeros_like(base_weight)
            for i in range(min(100, len(S_base))):
                u_base = U_base[:, i].reshape(-1, 1)
                v_base = Vh_base[i].reshape(1, -1)
                projection = u_base @ v_base
                # Project the LoRA update onto this direction
                projection_coef = np.sum(lora_update * projection)
                projection_matrix += projection_coef * projection
            
            # Compute residual (the part of the update that's orthogonal to the base space)
            residual = lora_update - projection_matrix
            residual_energy = np.linalg.norm(residual)**2
            projection_energy = np.linalg.norm(projection_matrix)**2
            
            # Compute relative energies
            relative_novel_energy = residual_energy / (residual_energy + projection_energy)
            relative_aligned_energy = projection_energy / (residual_energy + projection_energy)
            
            # Analyze alignments between LoRA singular vectors and base singular vectors
            alignments = []
            for i in range(num_vectors):
                # For each LoRA singular direction, compute alignment with base directions
                u_lora = U_lora[:, i]
                v_lora = Vh_lora[i]
                
                # Compute alignments with base singular vectors
                u_alignments = [np.abs(np.dot(u_lora, U_base[:, j])) for j in range(min(100, len(S_base)))]
                v_alignments = [np.abs(np.dot(v_lora, Vh_base[j])) for j in range(min(100, len(S_base)))]
                
                # Get maximum alignments
                max_u_alignment = max(u_alignments)
                max_v_alignment = max(v_alignments)
                
                # Use the minimum of the two (both need to align well)
                alignments.append(min(max_u_alignment, max_v_alignment))
            
            # Categorize alignments
            strongly_aligned = sum(1 for a in alignments if a > 0.9)
            moderately_aligned = sum(1 for a in alignments if 0.7 < a <= 0.9)
            weakly_aligned = sum(1 for a in alignments if 0.4 < a <= 0.7)
            novel_directions = sum(1 for a in alignments if a <= 0.4)
            
            # Analyze rank utilization
            rank_utilization = effective_rank / lora_rank if lora_rank else 0
            
            # Check singular value decay within the LoRA rank
            sv_ratio = S_lora[min(lora_rank-1, len(S_lora)-1)] / S_lora[0] if len(S_lora) > 0 else 0
            
            # Print concise analysis
            print(f"\nLoRA update analysis: ER={effective_rank}, E95={energy_95}, RM={relative_magnitude:.4f}")
            print(f"Alignments: Strong={strongly_aligned}, Mod={moderately_aligned}, Weak={weakly_aligned}, Novel={novel_directions}")
            print(f"Energy: Aligned={100*relative_aligned_energy:.1f}%, Novel={100*relative_novel_energy:.1f}%")
            
            return {
                "effective_rank": effective_rank,
                "energy_95": energy_95,
                "relative_magnitude": float(relative_magnitude),
                "strongly_aligned": strongly_aligned,
                "moderately_aligned": moderately_aligned,
                "weakly_aligned": weakly_aligned,
                "novel_directions": novel_directions,
                "relative_novel_energy": float(relative_novel_energy),
                "singular_values": S_lora.tolist()[:10],
                "alignments": alignments,
                "rank_utilization": rank_utilization,
                "singular_value_decay": sv_ratio,
                "projection_energy": float(relative_aligned_energy)
            }
            
        else:
            # For full fine-tuning, analyze the difference between fine-tuned and base weights
            weight_diff = weight_matrix - base_weight
            
            # Compute SVD of the difference
            U_diff, S_diff, Vh_diff = np.linalg.svd(weight_diff, full_matrices=False)
            
            # Relative magnitude of change
            update_norm = np.linalg.norm(weight_diff)
            base_norm = np.linalg.norm(base_weight)
            relative_magnitude = update_norm / base_norm
            
            # Analyze rank properties
            sv_threshold = 1e-5 * S_base[0]
            effective_rank = np.sum(S_diff > sv_threshold)
            
            # Energy analysis
            total_energy = np.sum(S_diff**2)
            energy_95 = np.searchsorted(np.cumsum(S_diff**2) / total_energy, 0.95) + 1
            
            # Project weight difference onto base model's singular vectors
            projection_matrix = np.zeros_like(base_weight)
            for i in range(min(100, len(S_base))):
                u_base = U_base[:, i].reshape(-1, 1)
                v_base = Vh_base[i].reshape(1, -1)
                projection = u_base @ v_base
                # Project the weight difference onto this direction
                projection_coef = np.sum(weight_diff * projection)
                projection_matrix += projection_coef * projection
            
            # Compute residual (the part of the update that's orthogonal to the base space)
            residual = weight_diff - projection_matrix
            residual_energy = np.linalg.norm(residual)**2
            projection_energy = np.linalg.norm(projection_matrix)**2
            
            # Compute relative energies
            relative_novel_energy = residual_energy / (residual_energy + projection_energy)
            relative_aligned_energy = projection_energy / (residual_energy + projection_energy)
            
            # Analyze alignments between difference singular vectors and base singular vectors
            num_vectors = min(100, len(S_diff))
            alignments = []
            for i in range(num_vectors):
                # For each difference singular direction, compute alignment with base directions
                u_diff = U_diff[:, i]
                v_diff = Vh_diff[i]
                
                # Compute alignments with base singular vectors
                u_alignments = [np.abs(np.dot(u_diff, U_base[:, j])) for j in range(min(100, len(S_base)))]
                v_alignments = [np.abs(np.dot(v_diff, Vh_base[j])) for j in range(min(100, len(S_base)))]
                
                # Get maximum alignments
                max_u_alignment = max(u_alignments)
                max_v_alignment = max(v_alignments)
                
                # Use the minimum of the two (both need to align well)
                alignments.append(min(max_u_alignment, max_v_alignment))
            
            # Categorize alignments
            strongly_aligned = sum(1 for a in alignments if a > 0.9)
            moderately_aligned = sum(1 for a in alignments if 0.7 < a <= 0.9)
            weakly_aligned = sum(1 for a in alignments if 0.4 < a <= 0.7)
            novel_directions = sum(1 for a in alignments if a <= 0.4)
            
            # Print concise analysis
            print(f"\nWeight diff analysis: ER={effective_rank}, E95={energy_95}, RM={relative_magnitude:.4f}")
            print(f"Alignments: Strong={strongly_aligned}, Mod={moderately_aligned}, Weak={weakly_aligned}, Novel={novel_directions}")
            print(f"Energy: Aligned={100*relative_aligned_energy:.1f}%, Novel={100*relative_novel_energy:.1f}%")
            
            return {
                "effective_rank": effective_rank,
                "energy_95": energy_95,
                "relative_magnitude": float(relative_magnitude),
                "strongly_aligned": strongly_aligned,
                "moderately_aligned": moderately_aligned,
                "weakly_aligned": weakly_aligned,
                "novel_directions": novel_directions,
                "relative_novel_energy": float(relative_novel_energy),
                "singular_values": S_diff.tolist()[:10],
                "alignments": alignments,
                "rank_utilization": 0.0,
                "singular_value_decay": 0.0,
                "projection_energy": float(relative_aligned_energy)
            }
    
    except Exception as e:
        print(f"Error in compute_weight_analysis: {e}")
        traceback.print_exc()
        return {
            "effective_rank": 0,
            "energy_95": 0,
            "relative_magnitude": 0.0,
            "strongly_aligned": 0,
            "moderately_aligned": 0,
            "weakly_aligned": 0,
            "novel_directions": 0,
            "relative_novel_energy": 0.0,
            "singular_values": [],
            "alignments": [],
            "rank_utilization": 0.0,
            "singular_value_decay": 0.0,
            "projection_energy": 0.0
        }

def analyze_layer_adaptation(weight_matrices, base_weights, lora_updates=None, is_lora=False, lora_rank=None):
    """Analyze how different layers adapt during training."""
    layer_metrics = {}
    
    if is_lora and lora_updates:
        print("\nAnalyzing LoRA layer adaptations...")
        # For LoRA, analyze the update matrices directly
        for layer_name, update in lora_updates.items():
            # Find the corresponding base weight
            base_weight_name = None
            for name in base_weights:
                if layer_name in name and 'weight' in name:
                    base_weight_name = name
                    break
            
            if not base_weight_name:
                print(f"Warning: Could not find base weight for {layer_name}, skipping...")
                continue
                
            base_weight = base_weights[base_weight_name]
            
            # Separate analysis for Q and V projections
            if 'q_proj' in layer_name:
                layer_type = 'query'
            elif 'v_proj' in layer_name:
                layer_type = 'value'
            else:
                layer_type = 'other'
                
            # Get layer index
            try:
                layer_idx = int(layer_name.split('.')[2]) if '.self_attn.' in layer_name else -1
            except (IndexError, ValueError):
                layer_idx = -1
                print(f"Warning: Could not extract layer index from {layer_name}")
            
            print(f"\nAnalyzing LoRA update for layer: {layer_name}")
            print(f"  Update shape: {update.shape}")
            print(f"  Base weight shape: {base_weight.shape}")
            
            metrics = compute_weight_analysis(
                update, base_weight,
                is_lora_update=True,
                lora_rank=lora_rank
            )
            
            layer_metrics[layer_name] = {
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                **metrics
            }
    else:
        print("\nAnalyzing full fine-tuning layer adaptations...")
        # For full fine-tuning, compare with base weights
        for layer_name, weight in weight_matrices.items():
            if layer_name not in base_weights:
                print(f"Warning: {layer_name} not found in base weights, skipping...")
                continue
                
            base_weight = base_weights[layer_name]
            
            # Separate analysis for Q and V projections
            if 'q_proj' in layer_name:
                layer_type = 'query'
            elif 'v_proj' in layer_name:
                layer_type = 'value'
            else:
                layer_type = 'other'
                
            # Get layer index
            try:
                layer_idx = int(layer_name.split('.')[2]) if '.self_attn.' in layer_name else -1
            except (IndexError, ValueError):
                layer_idx = -1
                print(f"Warning: Could not extract layer index from {layer_name}")
            
            print(f"\nAnalyzing weight changes for layer: {layer_name}")
            print(f"  Weight shape: {weight.shape}")
            print(f"  Base weight shape: {base_weight.shape}")
            
            metrics = compute_weight_analysis(
                weight, base_weight,
                is_lora_update=False
            )
            
            layer_metrics[layer_name] = {
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                **metrics
            }
    
    return layer_metrics

def analyze_singular_vector_similarity():
    """Analyze weight space changes across models and checkpoints."""
    try:
        results = []
        output_dir = "svd_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load pretrained model
        print("\nLoading pretrained model...")
        pretrained_model = load_model({"type": "pretrained", "task_type": "similar"})
        pretrained_weights = get_weight_matrices(pretrained_model)
        print(f"Extracted {len(pretrained_weights)} weight matrices from pretrained model")
        
        del pretrained_model
        torch.cuda.empty_cache()
        
        # Track progress across all models and checkpoints
        total_models = len(SIMILAR_TASK_CONFIGS)
        print(f"\nAnalyzing {total_models} models (both LoRA and full fine-tuning)...")
        
        # Analyze each model configuration
        model_configs = list(SIMILAR_TASK_CONFIGS.items())
        for model_idx, (model_name, config) in enumerate(model_configs, 1):
            try:
                # Get available checkpoint steps
                checkpoint_steps = get_checkpoint_steps(config["name"])
                if not checkpoint_steps:
                    print(f"No checkpoints found for {model_name}, skipping...")
                    continue
                
                # Show progress for models
                model_progress = f"[{model_idx}/{total_models}] Analyzing {model_name} ({config['type']})"
                print(f"\n{model_progress}")
                print("=" * len(model_progress))
                
                if config["type"] == "lora":
                    # For LoRA models, we'll load the model once and analyze it
                    # This is because we can only load the latest weights, not specific checkpoints
                    print(f"\nLoading LoRA model {model_name}...")
                    model = load_model(config, None)  # Pass None to load the latest weights
                    
                    # Get weight matrices and base weights
                    weight_matrices = get_weight_matrices(model)
                    print(f"Processing LoRA model with rank {config['rank']}")
                    
                    # Analyze each layer
                    for layer_name in pretrained_weights:
                        if layer_name in weight_matrices:
                            print(f"Analyzing layer: {layer_name}")
                            
                            # Get the weight matrices
                            pretrained_weight = pretrained_weights[layer_name]
                            lora_weight = weight_matrices[layer_name]
                            
                            # Check if this is a LoRA update matrix
                            is_lora_update = isinstance(lora_weight, tuple) and len(lora_weight) == 2
                            
                            if is_lora_update:
                                lora_update, base_weight = lora_weight
                                # Analyze the LoRA update
                                analysis = compute_weight_analysis(lora_update, base_weight, is_lora_update=True, lora_rank=config["rank"])
                                
                                # Store results for each layer, using the last checkpoint step for reference
                                last_step = checkpoint_steps[-1]
                                results.append({
                                    "model": model_name,
                                    "checkpoint": last_step,  # Use the last checkpoint step
                                    "layer": layer_name,
                                    "type": config["type"],
                                    "rank": config.get("rank"),
                                    "task_type": config.get("task_type", "unknown"),
                                    **analysis
                                })
                            else:
                                print(f"Skipping layer {layer_name} - not a LoRA update")
                    
                    # Clear GPU memory
                    del model
                    torch.cuda.empty_cache()
                else:
                    # Process each checkpoint for full fine-tuning models
                    total_checkpoints = len(checkpoint_steps)
                    for step_idx, step in enumerate(checkpoint_steps, 1):
                        try:
                            print(f"\nCheckpoint {step_idx}/{total_checkpoints}: {step}")
                            
                            # Load checkpoint
                            model = load_model(config, step)
                            
                            # For full fine-tuning, compare with pretrained weights
                            weight_matrices = get_weight_matrices(model)
                            print(f"Extracted {len(weight_matrices)} weight matrices from fine-tuned model at step {step}")
                            
                            # Analyze each layer
                            for layer_name in pretrained_weights:
                                if layer_name in weight_matrices:
                                    print(f"Analyzing layer: {layer_name}")
                                    
                                    # Get the weight matrices
                                    pretrained_weight = pretrained_weights[layer_name]
                                    ft_weight = weight_matrices[layer_name]
                                    
                                    # Analyze the weight difference
                                    analysis = compute_weight_analysis(ft_weight, pretrained_weight, is_lora_update=False)
                                    
                                    # Store results for each layer
                                    results.append({
                                        "model": model_name,
                                        "checkpoint": step,
                                        "layer": layer_name,
                                        "type": config["type"],
                                        "rank": config.get("rank"),
                                        "task_type": config.get("task_type", "unknown"),
                                        **analysis
                                    })
                            
                            # Clear GPU memory
                            del model
                            torch.cuda.empty_cache()
                        
                        except Exception as e:
                            print(f"Error processing checkpoint {step} for {model_name}: {e}")
                            traceback.print_exc()
                            continue
                
                # Save intermediate results
                if len(results) > 0:
                    intermediate_df = pd.DataFrame(results)
                    intermediate_path = os.path.join(output_dir, "similarity_results_intermediate.csv")
                    intermediate_df.to_csv(intermediate_path, index=False)
                    print(f"\nSaved intermediate results to: {intermediate_path}")
                    print(f"Current results shape: {intermediate_df.shape}")
                    print("\nModel types in results:")
                    print(intermediate_df['type'].value_counts())
            
            except Exception as e:
                print(f"Error processing model {model_name}: {e}")
                traceback.print_exc()
                continue
        
        # Create final DataFrame
        if len(results) > 0:
            results_df = pd.DataFrame(results)
            print("\nFinal results summary:")
            print(f"Total rows: {len(results_df)}")
            print("\nModel types:")
            print(results_df['type'].value_counts())
            print("\nUnique models:")
            print(results_df['model'].unique())
            
            # Save final results
            results_path = os.path.join(output_dir, "similarity_results.csv")
            results_df.to_csv(results_path, index=False)
            print(f"\nSaved final results to: {results_path}")
            
            return results_df, output_dir
        else:
            print("Warning: No results were collected. Creating empty DataFrame.")
            return pd.DataFrame(), output_dir
    
    except Exception as e:
        print(f"Error in analyze_singular_vector_similarity: {e}")
        traceback.print_exc()
        return pd.DataFrame(), "svd_analysis"

def plot_similarity_evolution(results_df, output_dir):
    """Create visualizations for weight space changes and layer adaptations."""
    try:
        # Create output directory for plots
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Try to load existing results if available
        results_path = os.path.join(output_dir, "similarity_results.csv")
        if os.path.exists(results_path):
            print(f"Loading existing results from {results_path}")
            results_df = pd.read_csv(results_path)
        
        # Print summary of available data
        print("\nData summary:")
        print(f"Total rows: {len(results_df)}")
        print("\nModels present:")
        for model in results_df['model'].unique():
            model_data = results_df[results_df['model'] == model]
            print(f"{model}: {len(model_data)} rows, {len(model_data['checkpoint'].unique())} checkpoints")
        
        # Normalize checkpoint steps for better visualization
        normalized_df = normalize_checkpoint_steps(results_df)
        
        # Get unique models and sort them by type and rank
        models = results_df['model'].unique()
        model_types = {model: results_df[results_df['model'] == model]['type'].iloc[0] for model in models}
        model_ranks = {model: results_df[results_df['model'] == model]['rank'].iloc[0] for model in models}
        
        # Sort models: first full fine-tuning, then LoRA by rank
        sorted_models = sorted(models, key=lambda m: (0 if model_types[m] == 'full_ft' else 1, 
                                                    model_ranks[m] if pd.notnull(model_ranks[m]) else 0))
        
        # Define colors and markers for different models
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_models)))
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
        
        # Create plots for both raw and normalized data
        for plot_type in ['raw', 'normalized']:
            df = results_df if plot_type == 'raw' else normalized_df
            x_col = 'checkpoint' if plot_type == 'raw' else 'training_percentage'
            x_label = 'Training Steps' if plot_type == 'raw' else 'Training Progress (%)'
            
            # 1. Plot effective rank evolution
            plt.figure(figsize=(12, 8))
            for i, model in enumerate(sorted_models):
                model_data = df[df['model'] == model]
                if model_data.empty:
                    continue
                
                # Group by checkpoint and compute mean effective rank
                grouped = model_data.groupby(x_col)['effective_rank'].mean().reset_index()
                
                linestyle = '-' if model_types[model] == 'full_ft' else '--'
                label = f"{model} (Full FT)" if model_types[model] == 'full_ft' else f"{model} (LoRA r={model_ranks[model]})"
                
                plt.plot(grouped[x_col], grouped['effective_rank'],
                        label=label,
                        color=colors[i], marker=markers[i % len(markers)],
                        linestyle=linestyle, linewidth=2, markersize=8)
            
            plt.xlabel(x_label, fontsize=14)
            plt.ylabel('Effective Rank', fontsize=14)
            plt.title('Evolution of Effective Rank', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'effective_rank_evolution_{plot_type}.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Plot energy_95 evolution
            plt.figure(figsize=(12, 8))
            for i, model in enumerate(sorted_models):
                model_data = df[df['model'] == model]
                if model_data.empty:
                    continue
                
                # Group by checkpoint and compute mean energy_95
                grouped = model_data.groupby(x_col)['energy_95'].mean().reset_index()
                
                linestyle = '-' if model_types[model] == 'full_ft' else '--'
                label = f"{model} (Full FT)" if model_types[model] == 'full_ft' else f"{model} (LoRA r={model_ranks[model]})"
                
                plt.plot(grouped[x_col], grouped['energy_95'],
                        label=label,
                        color=colors[i], marker=markers[i % len(markers)],
                        linestyle=linestyle, linewidth=2, markersize=8)
            
            plt.xlabel(x_label, fontsize=14)
            plt.ylabel('95% Energy Rank', fontsize=14)
            plt.title('Evolution of 95% Energy Rank', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'energy_95_evolution_{plot_type}.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Layer-wise analysis plots
            # Create separate plots for query and value projections
            for layer_type in ['query', 'value']:
                plt.figure(figsize=(12, 8))
                for i, model in enumerate(sorted_models):
                    model_data = df[(df['model'] == model) & (df['layer_type'] == layer_type)]
                    if model_data.empty:
                        continue
                    
                    # Group by checkpoint and layer index
                    layer_evolution = model_data.groupby(['checkpoint', 'layer_idx'])['relative_magnitude'].mean().reset_index()
                    
                    # Plot heatmap of changes across layers
                    plt.scatter(layer_evolution['layer_idx'], layer_evolution['relative_magnitude'],
                              label=f"{model} ({model_types[model]})",
                              color=colors[i], marker=markers[i % len(markers)], alpha=0.6)
                
                plt.xlabel('Layer Index', fontsize=14)
                plt.ylabel('Relative Magnitude of Changes', fontsize=14)
                plt.title(f'Layer-wise Changes ({layer_type.capitalize()} Projection)', fontsize=16)
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'layer_changes_{layer_type}_{plot_type}.png'),
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            # 4. LoRA-specific plots
            lora_models = [m for m in sorted_models if model_types[m] == 'lora']
            if lora_models:
                # Plot rank utilization vs. configured rank
                plt.figure(figsize=(12, 8))
                ranks = []
                utilizations = []
                
                for model in lora_models:
                    model_data = df[df['model'] == model]
                    if model_data.empty:
                        continue
                    
                    rank = model_ranks[model]
                    ranks.append(rank)
                    
                    # Get the last checkpoint data
                    last_checkpoint = model_data.groupby('checkpoint')['rank_utilization'].mean().iloc[-1]
                    utilizations.append(last_checkpoint)
                
                plt.scatter(ranks, utilizations, s=100)
                plt.plot([min(ranks), max(ranks)], [1, 1], 'r--', label='Perfect Utilization')
                plt.xlabel('Configured LoRA Rank', fontsize=14)
                plt.ylabel('Rank Utilization', fontsize=14)
                plt.title('LoRA Rank Utilization vs. Configured Rank', fontsize=16)
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'lora_rank_utilization_{plot_type}.png'),
                           dpi=300)
                plt.close()
                
                # Plot singular value decay
                plt.figure(figsize=(12, 8))
                decays = []
                
                for model in lora_models:
                    model_data = df[df['model'] == model]
                    if model_data.empty:
                        continue
                    
                    rank = model_ranks[model]
                    ranks.append(rank)
                    
                    # Get the last checkpoint data
                    last_checkpoint = model_data.groupby('checkpoint')['singular_value_decay'].mean().iloc[-1]
                    decays.append(last_checkpoint)
                
                plt.scatter(ranks, decays, s=100)
                plt.xlabel('Configured LoRA Rank', fontsize=14)
                plt.ylabel('Singular Value Decay', fontsize=14)
                plt.title('LoRA Singular Value Decay vs. Rank', fontsize=16)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'lora_sv_decay_{plot_type}.png'),
                           dpi=300)
                plt.close()
                
                # Plot projection energy evolution
                plt.figure(figsize=(12, 8))
                for i, model in enumerate(lora_models):
                    model_data = df[df['model'] == model]
                    if model_data.empty:
                        continue
                    
                    # Group by checkpoint
                    grouped = model_data.groupby(x_col)['projection_energy'].mean().reset_index()
                    
                    plt.plot(grouped[x_col], grouped['projection_energy'],
                            label=f"Rank {model_ranks[model]}",
                            color=colors[i], marker=markers[i % len(markers)])
                
                plt.xlabel(x_label, fontsize=14)
                plt.ylabel('Projection Energy', fontsize=14)
                plt.title('Evolution of LoRA Projection Energy', fontsize=16)
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'lora_projection_energy_{plot_type}.png'),
                           dpi=300)
                plt.close()
        
        print(f"Saved plots to {plots_dir}")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        traceback.print_exc()
        raise

def main():
    """Main function to run the analysis."""
    try:
        output_dir = "svd_analysis"
        results_path = os.path.join(output_dir, "similarity_results.csv")
        
        # Check if results file already exists
        if os.path.exists(results_path):
            print(f"\nFound existing results at: {results_path}")
            print("Loading existing results and creating visualizations...")
            results_df = pd.read_csv(results_path)
        else:
            print("Starting singular vector similarity analysis...")
            results_df, output_dir = analyze_singular_vector_similarity()
            
            # Save raw results
            results_df.to_csv(results_path, index=False)
            print(f"\nSaved raw results to: {results_path}")
        
        # Create visualization
        print("\nCreating visualization...")
        plot_similarity_evolution(results_df, output_dir)
        
        print("\nAnalysis complete!")
    
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()

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
            
            # If no checkpoints found, use standard intervals
            if not steps:
                steps = list(range(500, 5000, 500))
                if 4827 not in steps:
                    steps.append(4827)  # Add the final step if not already included
                print(f"\nNo checkpoints found, using standard intervals: {steps}")
        
        if steps:
            print(f"\nFound checkpoint steps: {sorted(steps)}")
        else:
            print("\nWarning: No valid checkpoints found in repository")
            print("Looking for:")
            if repo_name.endswith("FullFT"):
                print("  - Full fine-tuning: checkpoint-X/model.safetensors")
            else:
                print("  - LoRA: checkpoint-X/adapter_config.json")
        
        return sorted(steps)
    
    except Exception as e:
        print(f"Error inspecting repository {repo_name}: {e}")
        return []

def normalize_checkpoint_steps(results_df):
    """
    Normalize checkpoint steps for better visualization.
    
    This handles the different scales between LoRA and full fine-tuning checkpoints.
    """
    # Create a copy to avoid modifying the original
    df = results_df.copy()
    
    # Get unique model types
    model_types = df["type"].unique()
    
    # If we have both LoRA and full fine-tuning models
    if "lora" in model_types and "full_ft" in model_types:
        # Get max steps for each type
        lora_max = df[df["type"] == "lora"]["checkpoint"].max()
        full_ft_max = df[df["type"] == "full_ft"]["checkpoint"].max()
        
        # Create a normalized checkpoint column
        df["normalized_checkpoint"] = df.apply(
            lambda row: (row["checkpoint"] / lora_max) if row["type"] == "lora" 
                        else (row["checkpoint"] / full_ft_max), 
            axis=1
        )
        
        # Scale to percentage of training (0-100%)
        df["training_percentage"] = df["normalized_checkpoint"] * 100
    else:
        # If only one type, just use percentage of max
        max_step = df["checkpoint"].max()
        df["normalized_checkpoint"] = df["checkpoint"] / max_step
        df["training_percentage"] = df["normalized_checkpoint"] * 100
    
    return df

def load_model(config, step=None):
    """
    Load a model from Hugging Face Hub.
    
    Args:
        config: Model configuration dictionary
        step: Checkpoint step to load
    
    Returns:
        Loaded model
    """
    try:
        model_type = config.get("type", "")
        
        # For pretrained model
        if model_type == "pretrained":
            print("Loading pretrained LLaMA-1B model...")
            model = LlamaForCausalLM.from_pretrained(
                "meta-llama/llama-3.2-1B",
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            print("Pretrained model loaded successfully")
            return model
        
        # Get the model name from config
        model_name = config.get("name", "")
        if not model_name:
            raise ValueError("Model name not provided in config")
        
        # For LoRA models
        if model_type == "lora":
            lora_rank = config.get("rank", 4)
            print(f"Loading LoRA model {model_name} with rank {lora_rank}...")
            
            # First load the base model
            base_model = LlamaForCausalLM.from_pretrained(
                "meta-llama/llama-3.2-1B",
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Then try to load the LoRA adapter
            try:
                # If step is provided, try to load from checkpoint subfolder
                if step is not None:
                    checkpoint_path = f"{model_name}/checkpoint-{step}"
                    print(f"Loading LoRA adapter from checkpoint: {checkpoint_path}")
                    model = PeftModel.from_pretrained(
                        base_model,
                        checkpoint_path,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                else:
                    # Otherwise load from the root directory
                    print(f"Loading LoRA adapter from root: {model_name}")
                    model = PeftModel.from_pretrained(
                        base_model,
                        model_name,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                
                print("LoRA model loaded successfully")
                return model
            except Exception as e:
                print(f"Error loading LoRA adapter: {e}")
                traceback.print_exc()
                
                # Store base model weights for comparison
                base_model.base_weights = {}
                for name, param in base_model.named_parameters():
                    if "lora" not in name.lower():
                        base_model.base_weights[name] = param.detach().cpu().float().numpy()
                
                print("Returning base model with stored weights")
                return base_model
        
        # For full fine-tuning models
        elif model_type == "full_ft":
            # If step is provided, load from checkpoint subfolder
            if step is not None:
                checkpoint_path = f"{model_name}/checkpoint-{step}"
                print(f"Loading full fine-tuned model from checkpoint: {checkpoint_path}")
                model = LlamaForCausalLM.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                # Otherwise load from the root directory
                print(f"Loading full fine-tuned model from root: {model_name}")
                model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            print("Full fine-tuned model loaded successfully")
            return model
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
