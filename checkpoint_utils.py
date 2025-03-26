"""
Checkpoint Utilities

This module provides functions to retrieve checkpoint information from HuggingFace models
for both similar and dissimilar task models.
"""

from huggingface_hub import list_repo_files, hf_hub_download
from typing import List, Dict, Optional, Union
import logging
import torch
from safetensors.torch import load_file as safe_load
from transformers import (
    LlamaForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)
from peft import PeftModel, PeftConfig, TaskType, LoraConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/llama-3.2-1B", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Model configurations for similar tasks
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

def get_checkpoint_steps(repo_name: str, model_type: str) -> List[int]:
    """
    Get available checkpoint steps from a HuggingFace repository.
    
    Args:
        repo_name (str): Name of the HuggingFace repository
        model_type (str): Type of model ('full_ft' or 'lora')
    
    Returns:
        List[int]: Sorted list of available checkpoint steps
    """
    try:
        logger.info(f"Inspecting repository: {repo_name}")
        files = list_repo_files(repo_name)
        steps = []
        
        # For full fine-tuning models
        if model_type == "full_ft":
            for file in files:
                if "checkpoint-" in file and "model.safetensors" in file:
                    try:
                        step = int(file.split("/")[0].replace("checkpoint-", ""))
                        if step not in steps:
                            steps.append(step)
                    except ValueError:
                        continue
        
        # For LoRA models
        elif model_type == "lora":
            for file in files:
                if "checkpoint-" in file and "adapter_model.safetensors" in file:
                    try:
                        step = int(file.split("checkpoint-")[1].split("/")[0])
                        if step not in steps:
                            steps.append(step)
                    except ValueError:
                        continue
        
        if steps:
            steps.sort()
            logger.info(f"Found checkpoint steps: {steps}")
        else:
            logger.warning("No checkpoints found in repository")
            logger.info("Looking for:")
            if model_type == "full_ft":
                logger.info("  - Full fine-tuning: checkpoint-X/model.safetensors")
            else:
                logger.info("  - LoRA: checkpoint-X/adapter_model.safetensors")
        
        return sorted(steps)
    
    except Exception as e:
        logger.error(f"Error inspecting repository {repo_name}: {e}")
        return []

def get_model_checkpoints(model_config: Dict) -> Optional[List[int]]:
    """
    Get checkpoints for a specific model configuration.
    
    Args:
        model_config (Dict): Model configuration dictionary containing name and type
    
    Returns:
        Optional[List[int]]: List of checkpoint steps or None if error occurs
    """
    try:
        model_name = model_config.get("name")
        model_type = model_config.get("type")
        
        if not model_name or not model_type:
            logger.error("Invalid model configuration: missing name or type")
            return None
        
        return get_checkpoint_steps(model_name, model_type)
    
    except Exception as e:
        logger.error(f"Error getting model checkpoints: {e}")
        return None

def get_all_checkpoints(task_type: str = "similar") -> Dict[str, List[int]]:
    """
    Get checkpoints for all models of a specific task type.
    
    Args:
        task_type (str): Type of task ('similar' or 'dissimilar')
    
    Returns:
        Dict[str, List[int]]: Dictionary mapping model names to their checkpoint steps
    """
    configs = SIMILAR_TASK_CONFIGS if task_type == "similar" else DISSIMILAR_TASK_CONFIGS
    checkpoints = {}
    
    for model_name, config in configs.items():
        steps = get_model_checkpoints(config)
        if steps:
            checkpoints[model_name] = steps
    
    return checkpoints

def load_base_model(task_type: str = "similar", device_map: str = "cuda") -> Union[AutoModelForSequenceClassification, AutoModelForCausalLM]:
    """
    Load the base LLaMA model.
    
    Args:
        task_type (str): Type of task ('similar' or 'dissimilar')
        device_map (str): Device mapping strategy for model loading
    
    Returns:
        Union[AutoModelForSequenceClassification, AutoModelForCausalLM]: Base LLaMA model
    """
    try:
        logger.info("Loading base LLaMA-1B model...")
        
        if task_type == "similar":
            # For similar tasks (classification), use AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(
                "meta-llama/llama-3.2-1B",
                num_labels=3,
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=True
            )
        else:
            # For dissimilar tasks (generation), use AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/llama-3.2-1B",
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=True
            )
        
        logger.info(f"Base model loaded successfully for {task_type} tasks")
        return model
    
    except Exception as e:
        logger.error(f"Error loading base model: {e}")
        raise

def load_model_for_checkpoint(
    model_config: Dict,
    checkpoint_step: Optional[int] = None,
    device_map: str = "cuda"
) -> Optional[Union[AutoModelForSequenceClassification, AutoModelForCausalLM, PeftModel]]:
    """
    Load a model for a specific checkpoint step.
    """
    try:
        model_name = model_config.get("name")
        model_type = model_config.get("type")
        task_type = model_config.get("task_type")
        
        if not all([model_name, model_type, task_type]):
            logger.error("Invalid model configuration: missing name, type, or task_type")
            return None
        
        # For LoRA models
        if model_type == "lora":
            try:
                logger.info(f"Loading LoRA adapter from: {model_name}")
                
                # First load base model
                base_model = load_base_model(task_type, device_map)
                
                # Create LoRA config
                lora_config = LoraConfig(
                    r=model_config.get("rank", 16),
                    lora_alpha=model_config.get("rank", 16) * 2,
                    lora_dropout=0.1,
                    bias="none",
                    task_type=TaskType.SEQ_CLS if task_type == "similar" else TaskType.CAUSAL_LM,
                    target_modules=["q_proj", "v_proj"]
                )
                
                # First load the adapter from the main repo
                model = PeftModel.from_pretrained(
                    base_model,
                    model_name,
                    config=lora_config,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    is_trainable=False  # Set to False since we're loading for inference
                )
                
                # If checkpoint step is specified, load that specific checkpoint
                if checkpoint_step is not None:
                    checkpoint_path = f"{model_name}/checkpoint-{checkpoint_step}"
                    logger.info(f"Loading checkpoint from: {checkpoint_path}")
                    # Load the checkpoint weights
                    try:
                        # Download the adapter model file from the hub
                        adapter_path = hf_hub_download(
                            repo_id=model_name,
                            filename=f"checkpoint-{checkpoint_step}/adapter_model.safetensors",
                            repo_type="model"
                        )
                        # Load the state dict using safetensors
                        state_dict = safe_load(adapter_path)
                        
                        # Fix state dict keys to match PEFT's expected format
                        new_state_dict = {}
                        for key, value in state_dict.items():
                            if "lora_A.weight" in key or "lora_B.weight" in key:
                                # Convert lora_A.weight -> lora_A.default.weight
                                new_key = key.replace(".weight", ".default.weight")
                                new_state_dict[new_key] = value
                            else:
                                new_state_dict[key] = value
                        
                        # Load the modified state dict
                        model.load_state_dict(new_state_dict, strict=False)
                        logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
                    except Exception as e:
                        logger.error(f"Error loading checkpoint from {checkpoint_path}: {str(e)}")
                        # Continue with the base adapter if checkpoint loading fails
                        logger.info("Falling back to base adapter")
                
                return model
                
            except Exception as e:
                logger.error(f"Error in LoRA loading process: {str(e)}")
                return None
        
        # For full fine-tuned models
        elif model_type == "full_ft":
            try:
                logger.info(f"Loading full fine-tuned model from: {model_name}")
                
                # If checkpoint step is specified, download and load the specific checkpoint
                if checkpoint_step is not None:
                    try:
                        # Download the model file from the hub
                        model_path = hf_hub_download(
                            repo_id=model_name,
                            filename=f"checkpoint-{checkpoint_step}/model.safetensors",
                            repo_type="model"
                        )
                        logger.info(f"Loading checkpoint from: {model_path}")
                        
                        # Load the appropriate model class based on task type
                        if task_type == "similar":
                            # Create config for classification
                            config = AutoConfig.from_pretrained(
                                "meta-llama/llama-3.2-1B",
                                num_labels=3,
                                torch_dtype=torch.float16,
                                trust_remote_code=True
                            )
                            # Initialize model with config and move to device first
                            model = AutoModelForSequenceClassification.from_config(config).to(device_map)
                            # Load weights
                            state_dict = safe_load(model_path)
                            # Load state dict
                            model.load_state_dict(state_dict, strict=True)
                            logger.info("Successfully loaded classification model weights")
                        else:
                            # For causal LM, initialize model first
                            model = AutoModelForCausalLM.from_pretrained(
                                "meta-llama/llama-3.2-1B",
                                torch_dtype=torch.float16,
                                trust_remote_code=True
                            ).to(device_map)
                            # Load checkpoint weights
                            state_dict = safe_load(model_path)
                            
                            # Handle missing lm_head by tying weights with input embeddings
                            if "lm_head.weight" not in state_dict:
                                logger.info("LM head weights not found in checkpoint, tying with input embeddings")
                                # Get the input embeddings
                                input_embeddings = None
                                for key in state_dict:
                                    if "embed_tokens" in key:
                                        input_embeddings = state_dict[key]
                                        break
                                if input_embeddings is not None:
                                    state_dict["lm_head.weight"] = input_embeddings
                            
                            # Load state dict
                            model.load_state_dict(state_dict, strict=True)
                            logger.info("Successfully loaded causal LM model weights")
                        
                        logger.info(f"Successfully loaded checkpoint from {model_path}")
                        return model
                        
                    except Exception as e:
                        logger.error(f"Error loading checkpoint: {str(e)}")
                        return None
                
                # If no checkpoint specified, load the final model
                else:
                    if task_type == "similar":
                        model = AutoModelForSequenceClassification.from_pretrained(
                            model_name,
                            num_labels=3,
                            torch_dtype=torch.float16,
                            device_map=device_map,
                            trust_remote_code=True
                        )
                    else:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,
                            device_map=device_map,
                            trust_remote_code=True
                        )
                    
                    logger.info(f"Successfully loaded model from {model_name}")
                    return model
            
            except Exception as e:
                logger.error(f"Error loading full fine-tuned model: {str(e)}")
                return None
        
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None
    
    except Exception as e:
        logger.error(f"Error in load_model_for_checkpoint: {str(e)}")
        return None

def get_model_with_checkpoints(task_type: str = "similar") -> Dict[str, Dict]:
    """
    Get all models with their available checkpoints for a specific task type.
    
    Args:
        task_type (str): Type of task ('similar' or 'dissimilar')
    
    Returns:
        Dict[str, Dict]: Dictionary containing model configurations and their checkpoints
    """
    configs = SIMILAR_TASK_CONFIGS if task_type == "similar" else DISSIMILAR_TASK_CONFIGS
    models_info = {}
    
    for model_name, config in configs.items():
        steps = get_model_checkpoints(config)
        if steps:
            models_info[model_name] = {
                "config": config,
                "checkpoints": steps
            }
    
    return models_info

def load_checkpoint_model(
    task_type: str,
    model_name: str,
    checkpoint_step: Optional[int] = None,
    device_map: str = "cuda"
) -> Optional[Union[AutoModelForSequenceClassification, AutoModelForCausalLM, PeftModel]]:
    """
    Load a specific model checkpoint for a given task type and model name.
    
    Args:
        task_type (str): Type of task ('similar' or 'dissimilar')
        model_name (str): Name of the model configuration (e.g., 'full_ft', 'lora_16')
        checkpoint_step (Optional[int]): Specific checkpoint step to load, if None loads the latest
        device_map (str): Device mapping strategy for model loading
    
    Returns:
        Optional[Union[AutoModelForSequenceClassification, AutoModelForCausalLM, PeftModel]]: Loaded model or None if error occurs
    """
    configs = SIMILAR_TASK_CONFIGS if task_type == "similar" else DISSIMILAR_TASK_CONFIGS
    
    if model_name not in configs:
        logger.error(f"Model {model_name} not found in {task_type} task configurations")
        return None
    
    config = configs[model_name]
    return load_model_for_checkpoint(config, checkpoint_step, device_map)

def test_similar_task_model(model):
    """
    Test a similar task (classification) model by running inference on a test input.
    
    Args:
        model: The loaded model to test
    
    Returns:
        bool: True if test passes, False otherwise
    """
    try:
        # Prepare a test input
        test_text = "This movie was great, I really enjoyed it!"
        inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Check if outputs have the expected format
        if not hasattr(outputs, 'logits'):
            logger.error("Model outputs don't have logits - model may not be loaded correctly")
            return False
            
        logits = outputs.logits
        predictions = torch.nn.functional.softmax(logits, dim=-1)
        
        # Check if predictions have the expected shape (batch_size, num_classes)
        if predictions.shape != torch.Size([1, 3]):
            logger.error(f"Unexpected prediction shape: {predictions.shape}")
            return False
            
        logger.info(f"Model prediction probabilities: {predictions[0].tolist()}")
        logger.info("Similar task model test passed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error testing similar task model: {str(e)}")
        return False

def test_dissimilar_task_model(model):
    """
    Test a dissimilar task (generation) model by running inference on a test input.
    
    Args:
        model: The loaded model to test
    
    Returns:
        bool: True if test passes, False otherwise
    """
    try:
        # Prepare a test input
        test_text = "public class HelloWorld {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, World!\");\n    }\n}"
        inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=1,
                do_sample=False
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Check if we got a non-empty output
        if not generated_text or len(generated_text) < len(test_text):
            logger.error("Generated text is too short or empty")
            return False
            
        logger.info(f"Generated text: {generated_text}")
        logger.info("Dissimilar task model test passed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error testing dissimilar task model: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage for getting checkpoints
    similar_checkpoints = get_all_checkpoints("similar")
    print("\nSimilar Task Checkpoints:")
    for model, steps in similar_checkpoints.items():
        print(f"{model}: {steps}")
    
    dissimilar_checkpoints = get_all_checkpoints("dissimilar")
    print("\nDissimilar Task Checkpoints:")
    for model, steps in dissimilar_checkpoints.items():
        print(f"{model}: {steps}")
    
    # Example usage for loading and testing models
    print("\nLoading and testing models:")
    
    print("\nTesting Similar Task Model (LoRA):")
    # Load and test similar task LoRA model (classification)
    similar_lora = load_checkpoint_model("similar", "lora_16", checkpoint_step=similar_checkpoints["lora_16"][0])
    if similar_lora:
        print("Successfully loaded similar task LoRA model (classification)")
        if test_similar_task_model(similar_lora):
            print("Similar task model passed inference test")
        else:
            print("Similar task model failed inference test")
    
    print("\nTesting Dissimilar Task Model (LoRA):")
    # Load and test dissimilar task LoRA model (generation)
    dissimilar_lora = load_checkpoint_model("dissimilar", "lora_16", checkpoint_step=dissimilar_checkpoints["full_ft"][0])
    if dissimilar_lora:
        print("Successfully loaded dissimilar task LoRA model (generation)")
        if test_dissimilar_task_model(dissimilar_lora):
            print("Dissimilar task model passed inference test")
        else:
            print("Dissimilar task model failed inference test")
    
    # Clean up GPU memory
    del similar_lora
    del dissimilar_lora
    torch.cuda.empty_cache() 