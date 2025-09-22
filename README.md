```markdown
# 🧩 Internal States Leakage Project

This repository provides a **data-to-model pipeline** for detecting and training on internal state leakage.  
The workflow proceeds in **four main steps**:

1. **Generate** data  
2. **Evaluate** generated outputs  
3. **Label** the evaluated data  
4. **Train** a binary classification model

The automation is managed by a `Makefile`, making it easy to reproduce the entire process.

---

## 📂 Project Structure

```

.
├── Makefile
├── scripts/
│   ├── generate.py
│   ├── eval\_literal\_copying.py
│   ├── label.py
│   └── train.py
├── data/
├── prompts/
├── outputs/
├── scores/
└── labels/

````

---

## 🚀 Quick Start

Make sure you have **GNU Make** and **Python 3** installed.

Clone this repository and run:

```bash
make help
````

This displays all available targets.

---

## 🛠️ Makefile Targets

### 🔹 `help` (default)

Show all available commands.

```bash
make help
```

---

### 🔹 `generate`

Generate synthetic data using the specified model.

```bash
make generate
```

This runs:

```bash
/bin/python3 scripts/generate.py \
    --input_file data/data.literal.json \
    --prompt_file prompts/prompts.literal.format1.json \
    --output_file outputs/outputs.literal.prompt1.meta-llama/Meta-Llama-3.1-8B.greedy.json \
    --model meta-llama/Meta-Llama-3.1-8B \
    --n_instances 1000
```

---

### 🔹 `evaluate`

Evaluate the generated data for literal copying.

```bash
make evaluate
```

Command executed:

```bash
/bin/python3 scripts/eval_literal_copying.py \
    --input outputs/outputs.literal.prompt1.meta-llama/Meta-Llama-3.1-8B.greedy.json \
    --output scores/scores-literal-copying.literal.prompt1.meta-llama/Meta-Llama-3.1-8B.greedy.json
```

---

### 🔹 `label`

Label the evaluated data for training.

```bash
make label
```

Command executed:

```bash
/bin/python3 scripts/label.py \
    --input scores/scores-literal-copying.literal.prompt1.meta-llama/Meta-Llama-3.1-8B.greedy.json \
    --output labels/labels.literal.prompt1.meta-llama/Meta-Llama-3.1-8B.json
```

---

### 🔹 `train`

Train the binary classification model.

```bash
make train
```

Command executed:

```bash
/bin/python3 scripts/train.py \
    --input labels/labels.literal.prompt1.meta-llama/Meta-Llama-3.1-8B.json \
    --model_output outputs/trained_model.meta-llama/Meta-Llama-3.1-8B.bin
```

> **Note:** The Makefile includes the combined target `train & verification`,
> but you can simply run `make train` to trigger the training step.

---

### 🔹 `clean`

Remove all generated files.

```bash
make clean
```

This cleans up:

* `outputs/*.json`
* `scores/*.json`
* `labels/*.json`
* `outputs/*.bin`

---

## 💡 Tips

* Update the `MODEL` or `N_INSTANCES` variables in the `Makefile` to customize runs.
* Ensure all required Python packages are installed before running the commands.


```
```
