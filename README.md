# Fine-Tuning a Small Language Model with Lamini Docs

## Overview

This project demonstrates instruction fine-tuning of a small causal language model on the **Lamini Docs question-answer dataset**.

The workflow covers:

1. Loading and preparing a question-answer dataset.
2. Tokenizing the data.
3. Splitting the dataset into training and test sets.
4. Loading a pretrained Pythia language model.
5. Fine-tuning the model with Hugging Face Transformers.
6. Saving the fine-tuned model locally.
7. Testing the fine-tuned model on Lamini Docs questions.
8. Evaluating the fine-tuned model on the **ARC-Easy** benchmark.

The project was developed in Google Colab using a GPU runtime.

---

## Project Workflow

```text
Lamini Docs Dataset
        |
        v
Data Preparation
        |
        v
Tokenization
        |
        v
Train / Test Split
        |
        v
Pretrained Pythia Model
        |
        v
Hugging Face Trainer
        |
        v
Fine-Tuned Model
        |
        +----------------------+
        |                      |
        v                      v
Lamini Docs Evaluation     ARC-Easy Benchmark
        |                      |
        v                      v
Exact-Match Evaluation     Accuracy = 36.91%
```

---

## Model

The notebook's training configuration uses:

```text
Model: EleutherAI/pythia-70m
Maximum sequence length: 2048 tokens
```

> **Note:** Some later experimental cells in the notebook refer to `EleutherAI/pythia-410m`. The main fine-tuning configuration in the notebook uses **Pythia-70M**, so the README identifies Pythia-70M as the primary training model.

---

## Dataset

The main dataset is:

```text
lamini/lamini_docs
```

The dataset contains question-answer pairs related to Lamini documentation.

In the notebook, the resulting tokenized dataset contains:

```text
Training examples: 1260
Test examples:      140
Total examples:    1400
```

The local dataset files generated during preprocessing include:

```text
lamini_docs.jsonl
lamini_docs_processed.jsonl
```

The notebook also prepares an Alpaca-style dataset:

```text
alpaca_processed.jsonl
```

but the primary fine-tuning workflow uses the Lamini Docs dataset.

---

## Data Preparation

The question-answer data is converted into an instruction-style format and tokenized.

The tokenizer uses:

```python
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
tokenizer.pad_token = tokenizer.eos_token
```

The maximum sequence length used by the training configuration is:

```text
2048
```

The tokenized dataset contains:

```text
question
answer
input_ids
attention_mask
labels
```

---

## Fine-Tuning Configuration

The notebook performs a short demonstration fine-tuning run with:

| Parameter | Value |
|---|---:|
| Learning rate | `1e-5` |
| Maximum training steps | `3` |
| Epochs | `1` |
| Train batch size/device | `1` |
| Evaluation batch size/device | `1` |
| Gradient accumulation steps | `4` |
| Optimizer | `adafactor` |
| Warmup steps | `1` |
| Evaluation strategy | `steps` |
| Evaluation steps | `120` |
| Save steps | `120` |
| Logging steps | `1` |
| Gradient checkpointing | `False` |
| Maximum sequence length | `2048` |

The model is trained using Hugging Face's `Trainer`.

---

## Saved Model

The fine-tuned model is saved locally under:

```text
lamini_docs_3_steps/final/
```

Intermediate checkpoints are stored under:

```text
lamini_docs_3_steps/checkpoint-3/
```

The final model can be loaded using:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "lamini_docs_3_steps/final"
)
```

---

## Inference

The notebook defines an inference function that:

1. Tokenizes the input question.
2. Sends the tokens to the model.
3. Generates output tokens.
4. Decodes the generated tokens.
5. Removes the original prompt from the generated text.

Example:

```python
question = "What is Lamini?"

answer = inference(
    question,
    model,
    tokenizer
)

print(answer)
```

---

## Evaluation

### 1. Lamini Docs Evaluation

The notebook includes an exact-match evaluation:

```python
def is_exact_match(a, b):
    return a.strip() == b.strip()
```

The evaluation compares:

```text
Predicted answer
        vs.
Target answer
```

The notebook also loads:

```text
lamini/lamini_docs_evaluation
```

for additional evaluation exploration.

### 2. ARC-Easy Benchmark

The fine-tuned local model was evaluated using:

**EleutherAI LM Evaluation Harness**

Command:

```bash
lm-eval run \
    --model hf \
    --model_args pretrained=/content/lamini_docs_3_steps/final \
    --tasks arc_easy \
    --device cuda:0 \
    --batch_size 1
```

### Result

The fine-tuned model achieved:

```text
ARC-Easy Accuracy: 36.91%
Normalized Accuracy: 34.93%
Standard Error: ±0.0099
```

The ARC-Easy evaluation successfully processed:

```text
2376 contexts
9501 log-likelihood requests
```

### Interpretation

ARC-Easy has four answer choices, so random guessing is approximately:

```text
25%
```

The observed score of:

```text
36.91%
```

is above random-guessing performance.

However, this result should not be described as a strong general reasoning score. The model was fine-tuned primarily on Lamini documentation Q&A data, not on ARC reasoning data.

Also, a before/after comparison with the original pretrained model is required before claiming that fine-tuning improved ARC-Easy performance.

---

## Important Reproducibility Note

The notebook was originally written around an older Lamini/Transformers workflow. During execution in a current Colab environment, several compatibility issues were encountered.

Examples included:

- `datasets` version/API changes.
- `BasicModelRunner` accepting only the hard-coded example model names in the provided `llama.py`.
- Changes in `TrainingArguments`.
- Removal/incompatibility of the older `floating_point_ops()` usage.
- Compatibility issues with older Lamini demonstration cells.

For the final working workflow, the project uses the Hugging Face:

```text
AutoTokenizer
AutoModelForCausalLM
TrainingArguments
Trainer
```

workflow.

The old `BasicModelRunner` demonstration cells are not required for the final Hugging Face fine-tuning workflow.

---

## ARC-Easy Tokenizer Fix

During ARC-Easy evaluation, the tokenizer saved in the final directory was found to be incomplete:

```text
Vocab size: 2
Number of tokens: 0
```

This caused the evaluation harness to fail because it received an empty encoded context.

The tokenizer was replaced with the original Pythia tokenizer:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-410m"
)

tokenizer.pad_token = tokenizer.eos_token

tokenizer.save_pretrained(
    "/content/lamini_docs_3_steps/final"
)
```

The corrected tokenizer produced:

```text
Vocab size: 50254
```

and successfully tokenized ARC prompts.

After this correction, the ARC-Easy evaluation completed successfully.

> When reproducing the experiment, make sure the tokenizer matches the actual base model used for fine-tuning. The notebook's primary training configuration uses Pythia-70M; therefore, if reproducing the primary workflow exactly, use the corresponding Pythia-70M tokenizer rather than substituting a different Pythia model.

---

## Repository Structure

A recommended GitHub structure is:

```text
finetuned-llm/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── fine_tuning.ipynb
│
├── src/
│   ├── utilities.py
│   └── llama.py
│
├── data/
│   └── README.md
│
└── results/
    └── arc_easy_results.txt
```

### Do not commit

Avoid committing:

```text
API keys
.env files
Colab secrets
private credentials
large model checkpoints
```

The fine-tuned model directory can be several hundred MB or more, so it is generally better to store large model artifacts separately rather than committing them directly to a normal GitHub repository.

---

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

For Google Colab, the notebook can be run using a GPU runtime.

---

## Running the Project

### 1. Prepare the dataset

Use the notebook to load/process:

```text
lamini_docs.jsonl
```

and generate the processed training data.

### 2. Load the pretrained model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "EleutherAI/pythia-70m"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
```

### 3. Train

Configure `TrainingArguments` and create a Hugging Face `Trainer`.

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()
```

### 4. Save

```python
trainer.save_model("lamini_docs_3_steps/final")
```

### 5. Evaluate

Run the ARC-Easy benchmark:

```bash
lm-eval run \
    --model hf \
    --model_args pretrained=/path/to/final \
    --tasks arc_easy \
    --device cuda:0 \
    --batch_size 1
```

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face Hub
- Lamini Docs dataset
- EleutherAI Pythia
- LM Evaluation Harness
- Google Colab
- JSON Lines
- Pandas
- tqdm

---

## Limitations

1. The demonstrated fine-tuning run uses only a small number of training steps (`3`), so it is primarily an educational demonstration.
2. The model is small compared with modern large language models.
3. Fine-tuning on documentation Q&A data does not specifically optimize the model for ARC reasoning.
4. ARC-Easy performance alone does not measure documentation question-answer quality.
5. A pretrained-vs-fine-tuned comparison is required to determine whether fine-tuning improves general benchmark performance.
6. The notebook contains older Lamini demonstration code that may not work unchanged with current library versions.

---


