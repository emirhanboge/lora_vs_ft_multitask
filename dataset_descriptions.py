"""
Dataset descriptions

This module contains descriptions for all datasets used in TreeMark,
including their modifications and usage instructions.
"""

DATASET_DESCRIPTIONS = {
    "sst2": """# SST-2 (Modified for LLaMA 1B)
This dataset is a modified version of the **Stanford Sentiment Treebank 2 (SST-2)**, a binary classification dataset for sentiment analysis.

## **Modifications:**
- Labels were originally `"negative"` and `"positive"`, now converted to integers (`0` for negative, `1` for positive).
- Each example includes a task prefix: `Task: SST2 | Sentence: ...`
- The dataset has been **tokenized using the LLaMA-1B tokenizer**.
- Maximum sequence length is **128 tokens**.

## **Dataset Usage:**
```python
from datasets import load_dataset
dataset = load_dataset("emirhanboge/sst2_llama1b_modified")
""",

    "mnli": """# MNLI (Modified for LLaMA 1B)

This dataset is a modified version of the Multi-Genre Natural Language Inference (MNLI) dataset.

## **Modifications:**
- Labels were originally `"entailment"`, `"contradiction"`, and `"neutral"`, now converted to integers (`0, 1, 2`).
- The `premise` and `hypothesis` columns were renamed to `text1` and `text2`.
- Each example includes a task prefix: `Task: MNLI | Premise: ... Hypothesis: ...`
- Tokenized using the **LLaMA-1B tokenizer**.
- Maximum sequence length is **128 tokens**.

## **Dataset Usage:**
```python
from datasets import load_dataset
dataset = load_dataset("emirhanboge/mnli_llama1b_modified")
""",

    "qqp": """# QQP (Modified for LLaMA 1B)

This dataset is a modified version of the **Quora Question Pairs (QQP)** dataset for duplicate question classification.

## **Modifications:**
- Labels were originally `"duplicate"` and `"not duplicate"`, now converted to integers (`0` and `1`).
- The `question1` and `question2` columns were renamed to `text1` and `text2`.
- Each example includes a task prefix: `Task: QQP | Q1: ... Q2: ...`
- Tokenized using the **LLaMA-1B tokenizer**.
- Maximum sequence length is **128 tokens**.

## **Dataset Usage:**
```python
from datasets import load_dataset
dataset = load_dataset("emirhanboge/qqp_llama1b_modified")
""",

    "sst2_mnli_qqp": """# Multi-Task Dataset: SST-2 + MNLI + QQP (Modified for LLaMA 1B)

This dataset is a combination of **SST-2, MNLI, and QQP** for multi-task learning.

## **Modifications:**
- Each example includes a task prefix:
  - **SST-2:** `"Task: SST2 | Sentence: ..."`
  - **MNLI:** `"Task: MNLI | Premise: ... Hypothesis: ..."`
  - **QQP:** `"Task: QQP | Q1: ... Q2: ..."`
- Labels are standardized to integer format.
- Tokenized using the **LLaMA-1B tokenizer**.
- Maximum sequence length is **128 tokens**.

## **Dataset Usage:**
```python
from datasets import load_dataset
dataset = load_dataset("emirhanboge/sst2_mnli_qqp_llama1b_modified")
""",

    "codex_glue": """# CodeXGLUE Code-to-Text (Python Only, Modified for LLaMA 1B)

This dataset is a subset of **CodeXGLUE**, specifically the **code-to-text** dataset from Google, which focuses on generating natural language descriptions from Python code.

## **Modifications:**
- Only Python-related tasks and code samples are included.
- Each example includes a task prefix: `Task: CodeXGLUE | Code: ...`
- The dataset has been **tokenized using the LLaMA-1B tokenizer**.
- Maximum sequence length is **1024 tokens** to accommodate long code snippets.

## **Dataset Usage:**
```python
from datasets import load_dataset
dataset = load_dataset("emirhanboge/code_x_glue_ct_code_to_text_llama1b")
```
""",

    "rajpurkar/squad_v2": """# SQuAD v2 (Modified for LLaMA 1B)

This dataset is a modified version of the **Stanford Question Answering Dataset v2 (SQuAD v2)**, a reading comprehension benchmark.

## **Modifications:**
- Based on **SQuAD v2** (`rajpurkar/squad_v2`).
- Each example includes a task prefix: `Task: QA | Context: ... Question: ...`
- The dataset has been **tokenized using the LLaMA-1B tokenizer**.
- Maximum sequence length is **512 tokens**.

## **Dataset Usage:**
```python
from datasets import load_dataset
dataset = load_dataset("emirhanboge/squad_v2_llama1b_modified")
```
""",

    "cnn_dailymail": """# CNN/DailyMail (Version 3.0.0, Modified for LLaMA 1B)

This dataset consists of **news articles and their summaries**, originally designed for text summarization tasks.

## **Modifications:**
- Based on **CNN/DailyMail (Version 3.0.0)**.
- Each example includes a task prefix: `Task: Summarization | Article: ...`
- The dataset has been **tokenized using the LLaMA-1B tokenizer**.
- Maximum sequence length is **2048 tokens**.

## **Dataset Usage:**
```python
from datasets import load_dataset
dataset = load_dataset("emirhanboge/cnn_dailymail_llama1b_modified")
```
""",

    "qa_code_summarization": """# Multi-Task Dataset: QA + Code + Summarization (Modified for LLaMA 1B)

This dataset is a combination of **SQuAD v2, CNN/DailyMail, and CodeXGLUE Code-to-Text**, designed for multi-task learning across question answering, summarization, and code generation.

## **Modifications:**
- Includes examples from the following datasets:
  - **SQuAD v2** (`rajpurkar/squad_v2`): `"Task: QA | Context: ... Question: ..."`
  - **CNN/DailyMail** (`cnn_dailymail`): `"Task: Summarization | Article: ..."`
  - **CodeXGLUE Code-to-Text** (`code_x_glue_ct_code_to_text`): `"Task: CodeSumm | Code: ..."`
- Labels are formatted for consistency across tasks.
- The dataset has been **tokenized using the LLaMA-1B tokenizer**.
- Maximum sequence length is **2048 tokens** to accommodate longer texts from CNN/DailyMail.

## **Dataset Usage:**
```python
from datasets import load_dataset
dataset = load_dataset("emirhanboge/qa_code_summarization_llama1b_modified")
```
"""
} 