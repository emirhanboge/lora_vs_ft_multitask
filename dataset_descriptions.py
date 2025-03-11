"""
Dataset descriptions for README files when uploading to Hugging Face Hub.
"""

DATASET_DESCRIPTIONS = {
    "squad_v2": """# SQuAD v2 Dataset (Modified for LLaMA 1B)

This dataset contains the SQuAD v2 question answering dataset preprocessed for LLaMA 1B model training.
The dataset has been tokenized and formatted for causal language modeling.

## Format
- **input_ids**: Tokenized input sequences
- **attention_mask**: Attention mask for the input sequences
- **labels**: Tokenized target sequences
- **task**: Task identifier

## Usage
This dataset is designed to be used with the LLaMA 1B model for question answering tasks.
""",
    "code_to_text": """# Code-to-Text Dataset (Modified for LLaMA 1B)

This dataset contains code snippets and their corresponding natural language descriptions, preprocessed for LLaMA 1B model training.
The dataset has been tokenized and formatted for causal language modeling.

## Format
- **input_ids**: Tokenized input sequences
- **attention_mask**: Attention mask for the input sequences
- **labels**: Tokenized target sequences
- **task**: Task identifier

## Usage
This dataset is designed to be used with the LLaMA 1B model for code summarization tasks.
""",
    "cnn_dailymail": """# CNN/DailyMail Dataset (Modified for LLaMA 1B)

This dataset contains news articles and their summaries from CNN and Daily Mail, preprocessed for LLaMA 1B model training.
The dataset has been tokenized and formatted for causal language modeling.

## Format
- **input_ids**: Tokenized input sequences
- **attention_mask**: Attention mask for the input sequences
- **labels**: Tokenized target sequences
- **task**: Task identifier

## Usage
This dataset is designed to be used with the LLaMA 1B model for text summarization tasks.
""",
    "qa_code_summarization": """# Combined QA and Code Summarization Dataset (Modified for LLaMA 1B)

This dataset combines question answering (SQuAD v2) and code summarization tasks, preprocessed for LLaMA 1B model training.
The dataset has been tokenized and formatted for causal language modeling.

## Format
- **input_ids**: Tokenized input sequences
- **attention_mask**: Attention mask for the input sequences
- **labels**: Tokenized target sequences
- **task**: Task identifier

## Usage
This dataset is designed to be used with the LLaMA 1B model for multi-task learning.
""",
    "qa_code_summarization_cnn": """# Combined QA, Code Summarization, and Summarization Dataset (Modified for LLaMA 1B)

This dataset combines question answering (SQuAD v2), code summarization, and text summarization tasks, preprocessed for LLaMA 1B model training.
The dataset has been tokenized and formatted for causal language modeling.

## Format
- **input_ids**: Tokenized input sequences
- **attention_mask**: Attention mask for the input sequences
- **labels**: Tokenized target sequences
- **task**: Task identifier

## Usage
This dataset is designed to be used with the LLaMA 1B model for multi-task learning across diverse tasks.
"""
} 