"""
Global configuration variables for TreeMark.

This module contains all global constants, paths, and mappings used across the project.
"""

import os
import ast
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Helper function to parse string lists and dicts from env vars
def parse_env_value(value):
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer")

# Create directories if they don't exist
for dir_path in [DATA_DIR, OUTPUT_DIR, TOKENIZER_PATH]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# Model configurations
MODEL_NAME = "emirhanboge/LLaMA_1B_sst2_mnli_qqp_FullFT_"  # Base model to use
HF_USERNAME = "emirhanboge"  # Hugging Face username

# Dataset paths
DATASET_PATHS = {
    # Similar tasks (Classification)
    "sst2": "emirhanboge/sst2_llama1b_modified",
    "mnli": "emirhanboge/mnli_llama1b_modified",
    "qqp": "emirhanboge/qqp_llama1b_modified",
    "sst2_mnli_qqp": "emirhanboge/sst2_mnli_qqp_llama1b_modified",
    
    # Dissimilar tasks (Generation)
    "squad_v2": "emirhanboge/rajpurkar_squad_v2_llama1b_modified",
    "cnn_dailymail": "emirhanboge/cnn_dailymail_llama1b_modified",
    "codex_glue": "emirhanboge/codex_glue_llama1b_modified",
    "qa_code_summarization": "emirhanboge/qa_code_summarization_llama1b_modified",
}

# Model paths
MODEL_PATHS = {
    # Single task models
    "sst2_ft": "emirhanboge/LLaMA_1B_sst2_FullFT",
    "sst2_lora": "emirhanboge/LLaMA_1B_sst2_LoRA_16",
    "mnli_ft": "emirhanboge/LLaMA_1B_mnli_FullFT",
    "mnli_lora": "emirhanboge/LLaMA_1B_mnli_LoRA_16",
    "qqp_ft": "emirhanboge/LLaMA_1B_qqp_FullFT",
    "qqp_lora": "emirhanboge/LLaMA_1B_qqp_LoRA_16",
    
    # Multi-task models (Similar tasks)
    "similar_ft": "emirhanboge/LLaMA_1B_sst2_mnli_qqp_FullFT_",
    "similar_lora_4": "emirhanboge/LLaMA_1B_sst2_mnli_qqp_LoRA_4",
    "similar_lora_8": "emirhanboge/LLaMA_1B_sst2_mnli_qqp_LoRA_8",
    "similar_lora_16": "emirhanboge/LLaMA_1B_sst2_mnli_qqp_LoRA_16",
    "similar_lora_32": "emirhanboge/LLaMA_1B_sst2_mnli_qqp_LoRA_32",
    "similar_lora_64": "emirhanboge/LLaMA_1B_sst2_mnli_qqp_LoRA_64",
}

# Task definitions
SIMILAR_TASKS = ["sst2", "mnli", "qqp"]
DISSIMILAR_TASKS = ["squad_v2", "cnn_dailymail", "codex_glue"]

# Task labels
TASK_LABELS = {
    "sst2": 2,  # Binary: negative (0), positive (1)
    "mnli": 3,  # Three-way: entailment (0), contradiction (1), neutral (2)
    "qqp": 2,   # Binary: not duplicate (0), duplicate (1)
}

# Metric mappings
METRIC_MAPPING = {
    "sst2": "accuracy",
    "mnli": ["accuracy", "f1"],
    "qqp": ["accuracy", "f1"],
}

# Maximum sequence lengths
MAX_LENGTHS = {
    "sst2": 128,
    "mnli": 128,
    "qqp": 128,
    "squad_v2": 512,
    "cnn_dailymail": 2048,
    "codex_glue": 1024,
}

# Results and evaluation
RESULTS_FILE = os.path.expandvars(os.getenv('RESULTS_FILE', f'{DATA_DIR}/evaluation_results.txt'))

# Hugging Face configuration
HF_TOKEN = os.getenv('HF_TOKEN', '')  # This should be set in the .env file

# Task-specific configurations
TASK_LABELS = parse_env_value(os.getenv('TASK_LABELS', '''{
    "sst2": 2,
    "mnli": 3,
    "qqp": 2
}'''))

# Dataset paths
MULTI_TASK_PATH = os.path.expandvars(os.getenv('MULTI_TASK_PATH', f'{DATA_DIR}/sst2_mnli_qqp'))
QA_CODE_SUMM_PATH = os.path.expandvars(os.getenv('QA_CODE_SUMM_PATH', f'{DATA_DIR}/qa_code_summarization'))

# Metric mapping for evaluation
METRIC_MAPPING = {
    "mnli": "glue",
    "qqp": "glue", 
    "sst2": "glue"
} 