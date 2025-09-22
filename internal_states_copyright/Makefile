# Updated Makefile for Internal States Leakage Project
# This Makefile follows the pipeline: generate -> evaluate -> label -> train

# Variables
PYTHON := /bin/python3
SCRIPTS_DIR := scripts
DATA_DIR := data
PROMPTS_DIR := prompts
OUTPUTS_DIR := outputs
SCORES_DIR := scores
LABELS_DIR := labels
MODEL := meta-llama/Meta-Llama-3.1-8B
N_INSTANCES := 1000

# Phony targets
.PHONY: all generate evaluate label train clean help

# Default target
all: help

# Help target
help:
	@echo "Available targets:"
	@echo "  generate         - Generate data."
	@echo "  evaluate         - Evaluate generated data."
	@echo "  label            - Label evaluated data."
	@echo "  train            - Train binary classification model."
	@echo "  clean            - Clean up generated files."

# Step 1: Generate data
generate:
	$(PYTHON) $(SCRIPTS_DIR)/generate.py \
		--input_file $(DATA_DIR)/data.literal.json \
		--prompt_file $(PROMPTS_DIR)/prompts.literal.format1.json \
		--output_file $(OUTPUTS_DIR)/outputs.literal.prompt1.$(MODEL).greedy.json \
		--model $(MODEL) \
		--n_instances $(N_INSTANCES)

# Step 2: Evaluate generated data
evaluate:
	$(PYTHON) $(SCRIPTS_DIR)/eval_literal_copying.py \
		--input $(OUTPUTS_DIR)/outputs.literal.prompt1.$(MODEL).greedy.json \
		--output $(SCORES_DIR)/scores-literal-copying.literal.prompt1.$(MODEL).greedy.json

# Step 3: Label evaluated data
label:
	$(PYTHON) $(SCRIPTS_DIR)/label.py \
		--input $(SCORES_DIR)/scores-literal-copying.literal.prompt1.$(MODEL).greedy.json \
		--output $(LABELS_DIR)/labels.literal.prompt1.$(MODEL).json

# Step 4: Train binary classification model
train & verification:
	$(PYTHON) $(SCRIPTS_DIR)/train.py \
		--input $(LABELS_DIR)/labels.literal.prompt1.$(MODEL).json \
		--model_output $(OUTPUTS_DIR)/trained_model.$(MODEL).bin

# Clean up generated files
clean:
	rm -rf $(OUTPUTS_DIR)/*.json $(SCORES_DIR)/*.json $(LABELS_DIR)/*.json $(OUTPUTS_DIR)/*.bin



