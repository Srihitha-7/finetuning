# Fine-Tuning a Small Language Model with Lamini Docs

## Overview

This project demonstrates instruction fine-tuning of a small causal language model on the **Lamini Docs question-answer dataset**.

The workflow covers:

1. Loading and preparing a question-answer dataset.
2. Tokenizing the data.
3. Splitting the dataset into training and test sets.
4. Loading a pretrained Pythia language model.
5. Fine-tuning the model with Hugging Face Transformers (short demonstration run).
6. Comparing the pretrained baseline against a fully fine-tuned model
   (`lamini/lamini_docs_finetuned`) on both in-domain and out-of-domain evaluation.
7. Evaluating both models on the **ARC-Easy** benchmark to measure the effect of
   fine-tuning on general reasoning ability.

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
Pretrained Pythia Model (EleutherAI/pythia-70m)
        |
        +----------------------------+
        |                            |
        v                            v
Short local training run        Reference fine-tuned model
(3 steps, demo only)             (lamini/lamini_docs_finetuned)
                                       |
                    +------------------+------------------+
                    |                                     |
                    v                                     v
        Lamini Docs Evaluation                     ARC-Easy Benchmark
        (exact-match, in-domain)                (general reasoning, out-of-domain)
                    |                                     |
                    v                                     v
        Fine-tuned model wins                Pretrained: 36.70%
        on domain questions                  Fine-tuned: 31.23%
```

---

## Model

The notebook's training configuration uses:

```text
Model: EleutherAI/pythia-70m
Maximum sequence length: 2048 tokens
```

Two versions of "fine-tuned" appear in the notebook and it's important to distinguish them:

1. **Local demo checkpoint** — trained in-notebook for only 3 steps
   (`lamini_docs_3_steps/final`). This exists purely to illustrate the training loop
   and is *not* a meaningfully fine-tuned model — 3 steps is far too few to change
   model behavior noticeably.
2. **Reference fine-tuned model** — `lamini/lamini_docs_finetuned`, a properly
   fine-tuned checkpoint on the full dataset. **This is the model used for the actual
   pretrained-vs-fine-tuned comparison and for the results reported below.**

> **Note:** Some earlier experimental cells in the notebook referenced
> `EleutherAI/pythia-410m`. The primary training/evaluation configuration uses
> **Pythia-70M** throughout; the tokenizer used for evaluation must match this model.

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

The notebook also prepares an Alpaca-style dataset (`alpaca_processed.jsonl`), but the
primary fine-tuning and evaluation workflow uses the Lamini Docs dataset.

---

## Data Preparation

The question-answer data is converted into an instruction-style format and tokenized.

```python
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
tokenizer.pad_token = tokenizer.eos_token
```

Maximum sequence length: `2048`

The tokenized dataset contains: `question`, `answer`, `input_ids`, `attention_mask`, `labels`.

---

## Fine-Tuning Configuration (local demo run)

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

This run is a demonstration of the training loop only. All reported evaluation
results below use `lamini/lamini_docs_finetuned`, a properly fine-tuned checkpoint,
not this 3-step local run.

---

## Evaluation

### 1. Lamini Docs Evaluation (in-domain, exact-match)

```python
def is_exact_match(a, b):
    return a.strip() == b.strip()
```

Predicted vs. target answers were compared for both `EleutherAI/pythia-70m`
(pretrained) and `lamini/lamini_docs_finetuned` (fine-tuned) on the same held-out
test set. The fine-tuned model produces answers substantially closer to the target
Lamini docs answers than the raw pretrained model.

### 2. ARC-Easy Benchmark (out-of-domain, general reasoning)

```bash
lm-eval run \
    --model hf \
    --model_args pretrained=EleutherAI/pythia-70m \
    --tasks arc_easy \
    --device cuda:0 \
    --batch_size 1 \
    --output_path /content/results_pretrained/

lm-eval run \
    --model hf \
    --model_args pretrained=lamini/lamini_docs_finetuned \
    --tasks arc_easy \
    --device cuda:0 \
    --batch_size 1 \
    --output_path /content/results_finetuned/
```

### Results

| Model | ARC-Easy Accuracy |
|---|---:|
| Pretrained (`pythia-70m`) | 36.70% |
| Fine-tuned (`lamini_docs_finetuned`) | 31.23% |

ARC-Easy has four answer choices, so random-guessing accuracy is approximately 25%.
Both models score above chance, but the fine-tuned model scores **lower** than the
pretrained baseline on this general benchmark.

### Why did accuracy decrease after fine-tuning?

This is an expected result, known as **catastrophic forgetting**, not a bug:

- `lamini/lamini_docs_finetuned` was fine-tuned on a narrow, single-domain dataset
  (Lamini documentation Q&A).
- ARC-Easy tests general science/reasoning knowledge, unrelated to that domain.
- Fine-tuning a small model (70M parameters, limited capacity) on a narrow task shifts
  its weights toward that task's distribution, which can come at the cost of some
  general knowledge retained from pretraining.
- The net effect: the fine-tuned model improves on its target domain (Lamini docs
  Q&A, see the exact-match evaluation above) but regresses on unrelated general
  benchmarks like ARC-Easy.

This specialization/generalization trade-off is well documented in the fine-tuning
literature and is more pronounced with:
- very small base models,
- full-parameter fine-tuning (as opposed to parameter-efficient methods like LoRA),
- narrow, single-domain training data.

It should be reported as a genuine finding of this project, not treated as an error
to fix.

---

## Important Reproducibility Note

The notebook was originally written around an older Lamini/Transformers workflow.
Several compatibility issues were encountered when run in a current Colab
environment, including `datasets` version/API changes, `BasicModelRunner` accepting
only hard-coded example model names, changes in `TrainingArguments`, and removal of
older `floating_point_ops()` usage. The final working workflow uses the standard
Hugging Face `AutoTokenizer` / `AutoModelForCausalLM` / `TrainingArguments` /
`Trainer` stack. The old `BasicModelRunner` demonstration cells are not required for
the fine-tuning workflow and can be removed.

### Local checkpoint tokenizer issue (3-step demo run only)

During early experimentation, the tokenizer saved alongside the local 3-step
checkpoint (`lamini_docs_3_steps/final`) was found to be incomplete (vocab size of 2),
which caused evaluation to fail. This was specific to that local checkpoint and was
not an issue with `lamini/lamini_docs_finetuned`, whose tokenizer is complete and
correctly paired with the model. **The final reported ARC-Easy results use
`lamini/lamini_docs_finetuned` directly from the Hugging Face Hub and do not require
any tokenizer patching.**

---

## Repository Structure

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
    └── results_arc_easy.txt
```

### Do not commit

Avoid committing API keys, `.env` files, Colab secrets, private credentials, or large
model checkpoints. Store large model artifacts separately from the Git repository.

---

## Installation

```bash
pip install -r requirements.txt
```

For Google Colab, run the notebook using a GPU runtime.

---

## Running the Project

### 1. Prepare the dataset

Use the notebook to load/process `lamini_docs.jsonl` and generate the processed
training data.

### 2. Load the pretrained model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "EleutherAI/pythia-70m"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

pretrained_model = AutoModelForCausalLM.from_pretrained(model_name)
```

### 3. Train (local demo run)

```python
trainer = Trainer(
    model=pretrained_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()
```

### 4. Load the reference fine-tuned model for comparison

```python
finetuned_model = AutoModelForCausalLM.from_pretrained("lamini/lamini_docs_finetuned")
tokenizer = AutoTokenizer.from_pretrained("lamini/lamini_docs_finetuned")
```

### 5. Evaluate both models

Run exact-match evaluation on the Lamini docs test set for both models, and run the
ARC-Easy benchmark for both models (see commands above).

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

1. The local demonstration fine-tuning run uses only 3 training steps and is for
   illustrating the training loop only — it is not the model used for reported results.
2. The model is small (70M parameters) compared with modern LLMs, making it more
   susceptible to catastrophic forgetting during fine-tuning.
3. Fine-tuning on documentation Q&A data does not optimize the model for ARC-style
   reasoning, and in fact trades off against it.
4. ARC-Easy performance alone does not measure documentation question-answer quality;
   both the in-domain (exact-match) and out-of-domain (ARC-Easy) results are needed
   for a complete picture.

---

