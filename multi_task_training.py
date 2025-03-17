"""
Multi-task training script 

This script handles fine-tuning LLaMA models on multiple tasks simultaneously,
supporting both similar tasks (classification) and dissimilar tasks (generation).
"""

import os
import torch
import json
import logging
import argparse
import numpy as np
import random
import time
import sys
from dataclasses import asdict
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorForSeq2Seq,
    AutoConfig,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, concatenate_datasets, DatasetDict, Value, Dataset, Sequence
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
from config import (
    DATA_DIR,
    OUTPUT_DIR,
    TOKENIZER_PATH,
    TASK_LABELS,
    METRIC_MAPPING,
    SIMILAR_TASKS,
    DISSIMILAR_TASKS,
    DATASET_PATHS,
    MAX_LENGTHS,
    EXAMPLES_PER_TASK,
)
from training_configs import (
    TrainingConfig,
    SIMILAR_TASK_CONFIGS,
    DISSIMILAR_TASK_CONFIGS,
    get_config_name,
)
from data_preprocessing import (
    SimilarTaskDatasetHandler,
    DissimilarTaskDatasetHandler,
    setup_tokenizer,
)

from typing import Optional, Dict, List, Union, Any, Tuple


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Constants
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_training_args(output_dir, max_seq_length=2048, config=None):
    """
    Get training arguments for the trainer.
    
    Args:
        output_dir: The directory to save the model to.
        max_seq_length: The maximum sequence length for training.
        config: TrainingConfig object containing training parameters
        
    Returns:
        training_args: Training arguments for the trainer.
    """
    logger.info(f"Setting up training arguments with max sequence length of {max_seq_length}")
    
    batch_size = config.batch_size

    logger.info(f"Task type: {config.task_type}")
    logger.info(f"Save steps: {config.save_steps}")
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.epochs,
        eval_strategy="no",
        save_strategy="steps",
        logging_strategy="steps",
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_steps=config.save_steps,  # This will use 5600 for dissimilar tasks as defined in config
        logging_steps=config.logging_steps,
        optim="adamw_torch",
        report_to="none",
        save_total_limit=config.save_total_limit,
        push_to_hub=False,
        disable_tqdm=False,  # Enable progress bars
        # Performance optimizations
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        gradient_checkpointing=config.use_gradient_checkpointing,
        bf16=config.bf16,
        fp16=config.fp16,
        # Memory optimizations
        ddp_find_unused_parameters=False,
        # Additional optimizations
        group_by_length=True,  # Group similar length sequences together
        length_column_name="length",
        remove_unused_columns=True,
    )

class BaseTrainer:
    """Base class for training models."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize the trainer with the given configuration."""
        self.config = config
        self.tokenizer = setup_tokenizer(config.base_model)
        self.train_dataset = None
        self.eval_dataset = None
        
    def get_lora_config(self, task_type: str) -> Optional[LoraConfig]:
        """Get the LoRA configuration if LoRA is enabled."""
        if not self.config.use_lora:
            return None
        
        return LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=task_type,
            target_modules=["q_proj", "v_proj"],
        )

class SimilarTaskTrainer(BaseTrainer):
    """Trainer for similar tasks (classification)."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.dataset_handler = SimilarTaskDatasetHandler(self.tokenizer)
    
    def get_model(self):
        """Get the model for similar tasks."""
        num_labels = len(TASK_LABELS[self.config.tasks[0]])
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.base_model,
            num_labels=num_labels,
        )
        
        lora_config = self.get_lora_config(TaskType.SEQ_CLS)
        if lora_config:
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        return model
    
    def load_dataset(self):
        """Load and preprocess datasets for similar tasks."""
        self.train_dataset, self.eval_dataset = self.dataset_handler.load_and_preprocess(self.config.tasks)
    
    @staticmethod
    def compute_metrics(eval_pred):
        """Compute metrics for similar tasks."""
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=1)
        
        accuracy = evaluate.load("accuracy")
        return accuracy.compute(predictions=predictions, references=labels)
    
    def train(self):
        """Train the model on similar tasks."""
        if model_already_trained(get_config_name(self.config)):
            logger.info(f"Model {get_config_name(self.config)} already trained, skipping...")
            return
        
        logger.info(f"Training model on tasks: {self.config.tasks}")
        
        self.model = self.get_model()
        self.load_dataset()
        
        training_args = get_training_args(os.path.join(OUTPUT_DIR, get_config_name(self.config)), config=self.config)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            # eval_dataset=self.eval_dataset, # no evaluation in training
            compute_metrics=self.compute_metrics,
        )
        
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        # Save config
        with open(os.path.join(training_args.output_dir, "config.json"), "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        
        logger.info(f"Training complete. Model saved to {training_args.output_dir}")

class DissimilarTaskTrainer(BaseTrainer):
    """Trainer for dissimilar tasks."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.dataset_handler = DissimilarTaskDatasetHandler(self.tokenizer)
    
    def get_model(self):
        """Get the model for dissimilar tasks."""
        logger.info(f"Loading model {self.config.base_model}...")
        
        if torch.cuda.is_available():
            logger.info("CUDA available, clearing GPU cache")
            torch.cuda.empty_cache()
        
        max_length = 2048  # Increased for longer sequences
        logger.info(f"Using maximum sequence length of {max_length}")
        
        # Load model configuration
        logger.info("Loading model configuration")
        config = AutoConfig.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            use_cache=False,
            max_position_embeddings=max_length,
        )
        
        logger.info("Loading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            config=config,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        logger.info("Model loaded successfully")
        
        model.gradient_checkpointing_enable()
        
        if self.config.use_lora:
            logger.info(f"Setting up LoRA with rank {self.config.lora_rank}")
            lora_config = self.get_lora_config(TaskType.CAUSAL_LM)
            if lora_config:
                logger.info("Applying LoRA adapter to model")
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()
        else:
            logger.info("Using full fine-tuning (no LoRA)")
        
        return model
    
    def load_dataset(self):
        """Load the dataset for dissimilar tasks training."""
        self.train_dataset, self.eval_dataset = self.dataset_handler.load_and_preprocess(self.config.tasks)
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        logits, labels = eval_pred
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"loss": loss.item()}
    
    def train(self):
        """Train the model on dissimilar tasks."""
        if model_already_trained(get_config_name(self.config)):
            logger.info(f"Model {get_config_name(self.config)} already trained, skipping...")
            return
        
        logger.info(f"Starting training for model on tasks: {self.config.tasks}")
        
        # Get model and load dataset
        logger.info("Initializing model...")
        model = self.get_model()
        
        logger.info("Loading and preparing dataset...")
        self.load_dataset()
        
        # Configure tokenizer
        logger.info("Setting up tokenizer...")
        if self.tokenizer.pad_token is None:
            logger.info("Setting pad_token to eos_token since it was None")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        if self.tokenizer.pad_token_id < 0:
            logger.warning(f"Tokenizer has negative pad_token_id: {self.tokenizer.pad_token_id}")
            if self.tokenizer.eos_token_id >= 0:
                logger.info(f"Using eos_token_id ({self.tokenizer.eos_token_id}) as pad_token_id")
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                logger.info("Using 0 as pad_token_id since both pad and eos tokens are negative")
                self.tokenizer.pad_token_id = 0
        
        # Set up data collator with proper padding
        logger.info("Setting up data collator...")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            return_tensors="pt",
        )
        
        # Set up training arguments
        logger.info("Configuring training arguments...")
        training_args = get_training_args(
            os.path.join(OUTPUT_DIR, get_config_name(self.config)), 
            max_seq_length=2048,  # Increased for generation tasks
            config=self.config
        )
        
        logger.info("Initializing Trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train the model
        logger.info("Starting training process...")
        trainer.train()
        logger.info("Training completed successfully")
        
        # Save the model
        model_name = get_config_name(self.config)
        logger.info(f"Saving model to {OUTPUT_DIR}/{model_name}")
        trainer.save_model(OUTPUT_DIR + "/" + model_name)
        logger.info(f"Model saved successfully to {OUTPUT_DIR}/{model_name}")

def model_already_trained(model_name):
    """Check if a model has already been trained."""
    return os.path.exists(os.path.join(OUTPUT_DIR, model_name))

def run_training_configuration(config, model_type="llama-3.2-1b"):
    """Run a specific training configuration."""
    logger.info(f"Starting training configuration: {get_config_name(config)}")
    logger.info(f"Configuration details: {config}")
    
    # Check if model has already been trained
    output_dir = os.path.join(OUTPUT_DIR, get_config_name(config))
    if os.path.exists(os.path.join(output_dir, "pytorch_model.bin")) or \
       os.path.exists(os.path.join(output_dir, "adapter_model.bin")):
        logger.info(f"Model already exists at {output_dir}, skipping training")
        return
    
    # Determine trainer class based on task type
    if config.task_type == "similar":
        logger.info("Using SimilarTaskTrainer for training similar tasks")
        trainer = SimilarTaskTrainer(config)
    else:
        logger.info("Using DissimilarTaskTrainer for training dissimilar tasks")
        trainer = DissimilarTaskTrainer(config)
    
    # Load and validate datasets
    trainer.load_dataset()
    
    logger.info("Validating training dataset...")
    if not trainer.dataset_handler.validate_dataset(trainer.train_dataset):
        logger.warning("Training dataset has issues but proceeding with caution")
    
    logger.info("Validating evaluation dataset...")
    if not trainer.dataset_handler.validate_dataset(trainer.eval_dataset):
        logger.warning("Evaluation dataset has issues but proceeding with caution")
    
    # Train the model
    logger.info(f"Starting training on {len(trainer.train_dataset)} examples")
    trainer.train()
    
    logger.info(f"Training configuration {get_config_name(config)} completed successfully")

def main():
    """Main function to run training configurations."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train models on similar or dissimilar tasks")
    parser.add_argument("--similar", action="store_true", help="Train on similar tasks")
    parser.add_argument("--dissimilar", action="store_true", help="Train on dissimilar tasks")
    parser.add_argument("--lora", type=int, help="Use LoRA for training with specified rank")
    parser.add_argument("--full-finetune", action="store_true", help="Use full fine-tuning")
    parser.add_argument("--prepare-datasets", action="store_true", help="Prepare datasets without training")
    parser.add_argument("--upload", action="store_true", help="Upload models to Hugging Face Hub")
    
    args = parser.parse_args()
    
    # Log startup information
    logger.info("="*80)
    logger.info("Starting Multi-Task Training Framework")
    logger.info("="*80)
    
    # Log GPU information
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"CUDA device {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("CUDA not available - training will be VERY slow")
    
    # Determine configurations to run
    configs_to_run = []
    
    if args.similar:
        logger.info("Selected similar task training")
        if args.lora is not None:
            # Filter for LoRA configs with specified rank
            configs_to_run.extend([c for c in SIMILAR_TASK_CONFIGS if c.use_lora and c.lora_rank == args.lora])
        if args.full_finetune:
            configs_to_run.extend([c for c in SIMILAR_TASK_CONFIGS if not c.use_lora])
        if args.lora is None and not args.full_finetune:
            configs_to_run.extend(SIMILAR_TASK_CONFIGS)
            
    if args.dissimilar:
        logger.info("Selected dissimilar task training")
        if args.lora is not None:
            # Filter for LoRA configs with specified rank
            configs_to_run.extend([c for c in DISSIMILAR_TASK_CONFIGS if c.use_lora and c.lora_rank == args.lora])
        if args.full_finetune:
            configs_to_run.extend([c for c in DISSIMILAR_TASK_CONFIGS if not c.use_lora])
        if args.lora is None and not args.full_finetune:
            configs_to_run.extend(DISSIMILAR_TASK_CONFIGS)
            
    if not args.similar and not args.dissimilar:
        logger.info("No task type specified, using all configurations")
        if args.lora is not None:
            # Filter all configs for specified LoRA rank
            configs_to_run.extend([c for c in SIMILAR_TASK_CONFIGS + DISSIMILAR_TASK_CONFIGS 
                                 if c.use_lora and c.lora_rank == args.lora])
        elif args.full_finetune:
            configs_to_run.extend([c for c in SIMILAR_TASK_CONFIGS + DISSIMILAR_TASK_CONFIGS if not c.use_lora])
        else:
            configs_to_run = SIMILAR_TASK_CONFIGS + DISSIMILAR_TASK_CONFIGS
    
    # Print summary of configs
    logger.info(f"Will run {len(configs_to_run)} configurations:")
    for idx, config in enumerate(configs_to_run):
        logger.info(f"  Config {idx+1}: {get_config_name(config)}")
        logger.info(f"    - Tasks: {config.tasks}")
        logger.info(f"    - Model: {config.base_model}")
        logger.info(f"    - Method: {'LoRA' if config.use_lora else 'Full Fine-tuning'}")
        if config.use_lora:
            logger.info(f"    - LoRA rank: {config.lora_rank}")
        logger.info(f"    - Batch size: {config.batch_size}")
        logger.info(f"    - Epochs: {config.epochs}")
        logger.info(f"    - Learning rate: {config.learning_rate}")
    
    # Run selected configurations
    logger.info(f"Starting training for {len(configs_to_run)} configurations")
    for i, config in enumerate(configs_to_run):
        logger.info(f"Configuration {i+1}/{len(configs_to_run)}: {get_config_name(config)}")
        run_training_configuration(config)
    
    # Upload models if requested
    if args.upload:
        logger.info("Uploading trained models to Hugging Face Hub")
        for config in configs_to_run:
            model_name = get_config_name(config)
            model_path = os.path.join(OUTPUT_DIR, model_name)
            if os.path.exists(model_path):
                logger.info(f"Uploading model {model_name} to Hub")
                if config.task_type == "similar":
                    handler = SimilarTaskDatasetHandler(None)
                else:
                    handler = DissimilarTaskDatasetHandler(None)
                handler.upload_model_to_hub(
                    model_name=model_name,
                    tasks=config.tasks,
                    model_path=model_path,
                    lora_rank=config.lora_rank if config.use_lora else None
                )
            else:
                logger.warning(f"Model {model_name} directory not found, skipping upload")
    
    logger.info("="*80)
    logger.info("Multi-Task Training complete!")
    logger.info("="*80)

if __name__ == "__main__":
    main()