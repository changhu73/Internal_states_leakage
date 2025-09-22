# Internal states leakage

```markdown
# 🧩 Internal States Leakage Project – Makefile Guide

This guide explains how to use the provided **Makefile** to run the full pipeline for the *Internal States Leakage* project.  
The workflow is: **generate ➜ evaluate ➜ label ➜ train & verification**.

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

## ⚡ Quick Start

Clone the repository and move into the project directory:

```bash
git clone <your-repo-url>
cd internal-states-leakage
````

Check the available targets:

```bash
make help
```

---

## 🛠️ Makefile Targets

| Target     | Description                                       |
| ---------- | ------------------------------------------------- |
| `generate` | Generate synthetic data using the specified model |
| `evaluate` | Evaluate generated data for literal copying       |
| `label`    | Assign labels based on evaluation scores          |
| `train`    | Train the binary classification model             |
| `clean`    | Remove all generated files                        |

---

## 🚀 Usage Steps

### 1️⃣ Generate Data

Produce outputs using the given model:

```bash
make generate
```

### 2️⃣ Evaluate

Evaluate generated outputs for literal copying:

```bash
make evaluate
```

### 3️⃣ Label

Create labeled data from evaluation scores:

```bash
make label
```

### 4️⃣ Train the Model

Train a binary classification model:

```bash
make train
```

---

## 🧹 Cleaning Up

Remove all generated JSON and model binary files:

```bash
make clean
```

---

## ⚙️ Key Variables

The Makefile defines the following important variables (edit them as needed):

```makefile
PYTHON := /bin/python3
MODEL := meta-llama/Meta-Llama-3.1-8B
N_INSTANCES := 1000
```

* **PYTHON**: Path to Python 3 interpreter
* **MODEL**: Model used for generation
* **N\_INSTANCES**: Number of data instances to generate

Modify these directly in the `Makefile` to customize runs.

---

## 💡 Tips

* Run `make help` anytime to list all targets and their descriptions.
* Ensure the `scripts/` directory contains all Python scripts before running.
* Adjust the `MODEL` variable if you want to experiment with other models.

```

