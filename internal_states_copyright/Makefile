# Makefile for Internal States Leakage Project
# This Makefile provides commands for generating data, evaluating models, and summarizing results.

# Variables
PYTHON := /bin/python3
SCRIPTS_DIR := scripts
DATA_DIR := data
PROMPTS_DIR := prompts
OUTPUTS_DIR := outputs
SCORES_DIR := scores
MODEL := meta-llama/Meta-Llama-3.1-8B
N_INSTANCES := 1000

# Phony targets
.PHONY: all evaluate eval_qa generate_literal clean help

# Default target
all: help

# Help target
help:
	@echo "Available targets:"
	@echo "  evaluate         - Evaluate literal copying."
	@echo "  eval_qa          - Evaluate QA performance."
	@echo "  generate_literal - Generate literal data."
	@echo "  clean            - Clean up generated files."

# Evaluate literal copying
evaluate:
	$(PYTHON) $(SCRIPTS_DIR)/eval_literal_copying.py \
		--input $(OUTPUTS_DIR)/outputs.literal.prompt1.$(MODEL).greedy.json \
		--output $(SCORES_DIR)/scores-literal-copying.literal.prompt1.$(MODEL).greedy.json

# Evaluate QA performance
eval_qa:
	$(PYTHON) $(SCRIPTS_DIR)/eval_qa.py \
		--input $(OUTPUTS_DIR)/outputs.qa.prompt1.$(MODEL).greedy.json \
		--output $(SCORES_DIR)/scores-qa.qa.prompt1.$(MODEL).greedy.json

# Generate literal data
generate_literal:
	$(PYTHON) $(SCRIPTS_DIR)/generate.py \
		--input_file $(DATA_DIR)/data.literal.json \
		--prompt_file $(PROMPTS_DIR)/prompts.literal.format1.json \
		--output_file $(OUTPUTS_DIR)/outputs.literal.prompt1.$(MODEL).greedy.json \
		--model $(MODEL) \
		--n_instances $(N_INSTANCES)

# Clean up generated files
clean:
	rm -rf $(OUTPUTS_DIR)/*.json $(SCORES_DIR)/*.json




/bin/python3 scripts/generate.py --input_file data/data.literal.json --prompt_file /prompts/prompts.literal.format1.json --output_file  outputs/outputs.literal.prompt1.Meta-Llama-3.1-8B.greedy.json --model meta-llama/Meta-Llama-3.1-8B --n_instances 758
/bin/python3 scripts/generate.py --input_file data/data.nonliteral.json --prompt_file /prompts/prompts.nonliteral.format1.json --output_file  outputs/outputs.nonliteral.prompt1.Meta-Llama-3.1-8B.greedy.json --model meta-llama/Meta-Llama-3.1-8B --n_instances 590
/bin/python3 scripts/generate.py --input_file data/data.qa.json --prompt_file /prompts/prompts.qa.format1.json --output_file  outputs/outputs.qa.prompt1.Meta-Llama-3.1-8B.greedy.json --model meta-llama/Meta-Llama-3.1-8B --n_instances 589

evaluate:
python scripts/eval_literal_copying.py --input  outputs/outputs.literal.prompt1.Meta-Llama-3.1-8B.greedy.json --output  scores/scores-literal-copying.literal.prompt1.Meta-Llama-3.1-8B.greedy.json
python scripts/generate.py --input_file data/data.literal.json --prompt_file /prompts/prompts.literal.format2.json --output_file /home/Guangwei/copy-bench/outputs/outputs.literal.prompt2.Meta-Llama-3.1-8B.greedy.json --model meta-llama/Meta-Llama-3.1-8B --n_instances 758

eval:
python scripts/eval_literal_copying.py --input outputs/outputs.literal.prompt1.Llama-3.1-70B.greedy.json --output  scores/scores-literal-copying.literal.prompt1.Llama-3.1-70B.greedy.json

export PYTHONPATH=./src:$$PYTHONPATH

meta-llama/Llama-2-70b-chat-hf

quick_start:
	$(eval HF_MODEL=meta-llama/Meta-Llama-3.1-8B)
	$(eval MODEL_TAG=Meta-Llama-3.1-8B)
	$(eval N=10)

	### Evaluating literal copying

	python scripts/generate.py \
		--input_file data/data.literal.json \
		--prompt_file prompts/prompts.literal.format1.json \
		--output_file outputs/outputs.literal.prompt1.$(MODEL_TAG).greedy.json \
		--model $(HF_MODEL) \
		--n_instances 758 \

	python scripts/generate.py --input_file data/data.literal.json --prompt_file prompts/prompts.literal.format1.json --output_file outputs/outputs.literal.prompt1.Meta-Llama-3.1-8B.greedy.json --model meta-llama/Meta-Llama-3.1-8B --n_instances 758

	python scripts/eval_literal_copying.py\
		--input outputs/outputs.literal.prompt1.$(MODEL_TAG).greedy.json \
		--output scores/scores-literal-copying.literal.prompt1.$(MODEL_TAG).greedy.json

	# python scripts/eval_quality.py \
	# 	--input outputs/outputs.literal.prompt1.$(MODEL_TAG).greedy.json \
	# 	--output scores/scores-quality.literal.prompt1.$(MODEL_TAG).greedy.json

	#### Evaluate non-literal copying
	python scripts/generate.py \
		--input_file /data/data.nonliteral.json \
		--prompt_file /prompts/prompts.nonliteral.format1.json \
		--output_file outputs/outputs.nonliteral.prompt1.$(MODEL_TAG).greedy.json \
		--model $(HF_MODEL) \
		--max_new_tokens 1024 \
		--n_instances 1000 \

	/bin/python3 scripts/generate.py --input_file data/data.nonliteral.json --prompt_file prompts/prompts.nonliteral.format1.json --output_file  outputs/outputs.nonliteral.prompt1.Meta-Llama-3.1-8B.greedy.json --model meta-llama/Meta-Llama-3.1-8B --n_instances 1000

	/home/Guangwei/.conda/envs/sit/bin/python /home/Guangwei/sit/copy-bench/scripts/eval_char_copying.py \
		--input outputs/outputs.nonliteral.prompt1.$(MODEL_TAG).greedy.json \
		--output scores/scores-char-copying.nonliteral.prompt1.$(MODEL_TAG).greedy.json

	/home/Guangwei/.conda/envs/sit/bin/python /home/Guangwei/sit/copy-bench/scripts/eval_event_copying.py \
		--input outputs/outputs.nonliteral.prompt1.$(MODEL_TAG).greedy.json \
		--output scores/scores-event-copying.nonliteral.prompt1.$(MODEL_TAG).greedy.json

	# python scripts/eval_quality.py \
	# 	--input outputs/outputs.nonliteral.prompt1.$(MODEL_TAG).greedy.json \
	# 	--output scores/scores-quality.nonliteral.prompt1.$(MODEL_TAG).greedy.json

	#### Evaluate Fact Recall
	python scripts/generate.py \
		--input_file data/data.qa.json \
		--prompt_file prompts/prompts.qa.format1.json \
		--output_file outputs/outputs.qa.prompt1.Meta-Llama-3.1-8B.greedy.json \
		--model meta-llama/Meta-Llama-3.1-8B \
		--n_instances 589 \

	 python  scripts/generate.py 
	--input_file  data/data.qa.json 
	--prompt_file   prompts/prompts.qa.format1.json --output_file  outputs/outputs.qa.prompt1.Meta-Llama-3.1-8B.greedy.json --model meta-llama/Meta-Llama-3.1-8B --n_instances 10000

	python scripts/eval_qa.py \
		--input outputs/outputs.qa.prompt1.$(MODEL_TAG).greedy.json \
		--output scores/scores-qa.qa.prompt1.$(MODEL_TAG).greedy.json

eval_qa:
 python  scripts/eval_qa.py --input outputs/outputs.qa.prompt1.Llama-3.1-70B.greedy.json --output scores/scores-qa.qa.prompt1.Llama-3.1-70B.greedy.json

python scripts/summary.py --root ./scores/**.json

generate_literal:
 python  scripts/generate.py --input_file data/data.extra.json --prompt_file   prompts/prompts.literal.format1.json --output_file  outputs/outputs.literal.prompt1.Llama-3.1-8B.greedy.json --model meta-llama/Meta-Llama-3.1-8B --n_instances 10000
 python  scripts/generate.py --input_file data/data.qa.json --prompt_file   prompts/prompts.qa.format1.json --output_file  outputs/outputs.qa.prompt1.Llama-3.1-70B.greedy.json --model meta-llama/Llama-3.1-70B --n_instances 10000
 python  scripts/generate.py --input_file data/data.nonliteral.json --prompt_file   prompts/prompts.nonliteral.format1.json --output_file  outputs/outputs.nonliteral.prompt1.Llama-3.1-70B.greedy.json --model meta-llama/Llama-3.1-70B --n_instances 10000


non:
 python scripts/generate.py --input_file data/data.nonliteral.json --prompt_file prompts/prompts.nonliteral.format1.json --output_file  outputs/outputs.nonliteral.prompt1.Meta-Llama-3.1-8B.greedy.json --model meta-llama/Meta-Llama-3.1-8B --n_instances 10000
 python scripts/eval_char_copying.py --input outputs/outputs.nonliteral.prompt1.Meta-Llama-3.1-8B.greedy.json --output scores/scores-char-copying.nonliteral.prompt1.Meta-Llama-3.1-8B.greedy.json

 python scripts/eval_event_copying.py --input outputs/outputs.nonliteral.prompt1.Meta-Llama-3.1-8B.greedy.json --output scores/scores-event-copying.nonliteral.prompt1.Meta-Llama-3.1-8B.greedy.json







 python  scripts/generate.py --input_file  data/data.literal.json --prompt_file   prompts/prompts.literal.format1.json --output_file  new_outputs/outputs.literal.prompt1.Llama-3.1-70B.greedy.json --model /raid/data/guangwei/huggingface/hub/models--meta-llama--Llama-3.1-70B/snapshots/349b2ddb53ce8f2849a6c168a81980ab25258dac/ --n_instances 10000

 python  scripts/eval_literal_copying.py --input  new_outputs/outputs.literal.prompt1.Llama-3.1-70B.greedy.json --output  new_scores/scores-literal-copying.literal.prompt1.Llama-3.1-70B.greedy.json


 python  scripts/generate.py --input_file  data/data.nonliteral.json --prompt_file  prompts/prompts.nonliteral.format1.json --output_file  new_outputs/outputs.nonliteral.prompt1.Llama-3.1-70B.greedy.json --model /raid/data/guangwei/huggingface/hub/models--meta-llama--Llama-3.1-70B/snapshots/349b2ddb53ce8f2849a6c168a81980ab25258dac/ --max_new_tokens 1024

 python  scripts/eval_char_copying.py --input  new_outputs/outputs.nonliteral.prompt1.Llama-3.1-70B.greedy.json --output  new_scores/scores-char-copying.nonliteral.prompt1.Llama-3.1-70B.greedy.json

 python  scripts/eval_event_copying.py --input  new_outputs/outputs.nonliteral.prompt1.Llama-3.1-70B.greedy.json --output  new_scores/scores-event-copying.nonliteral.prompt1.Llama-3.1-70B.greedy.json


 python  scripts/generate.py --input_file  data/data.qa.json --prompt_file  prompts/prompts.qa.format1.json --output_file  new_outputs/outputs.qa.prompt1.Llama-3.1-70B.greedy.json --model /raid/data/guangwei/huggingface/hub/models--meta-llama--Llama-3.1-70B/snapshots/349b2ddb53ce8f2849a6c168a81980ab25258dac/

 python  scripts/eval_qa.py --input  new_outputs/outputs.qa.prompt1.Llama-3.1-70B.greedy.json --output  new_scores/scores-qa.qa.prompt1.Llama-3.1-70B.greedy.json

 python  scripts/eval_fluency.py --input  new_outputs/outputs.literal.prompt1.Llama-3.1-70B.greedy.json --output  new_scores/scores-fluency.literal.prompt1.Llama-3.1-70B.greedy.json

 python  scripts/eval_fluency.py --input  new_outputs/outputs.nonliteral.prompt1.Llama-3.1-70B.greedy.json --output  new_scores/scores-fluency.nonliteral.prompt1.Llama-3.1-70B.greedy.json


 python  scripts/summary.py --root  new_scores/scores-literal-copying.literal.prompt1.Llama-3.1-70B.greedy.json  new_scores/scores-char-copying.nonliteral.prompt1.Llama-3.1-70B.greedy.json  new_scores/scores-event-copying.nonliteral.prompt1.Llama-3.1-70B.greedy.json  new_scores/scores-qa.qa.prompt1.Llama-3.1-70B.greedy.json  new_scores/scores-fluency.literal.prompt1.Llama-3.1-70B.greedy.json  new_scores/scores-fluency.nonliteral.prompt1.Llama-3.1-70B.greedy.json









new raid:

 python /raid/data/guangwei/copyright_newVersion/scripts/copybench/Llama_3.1_70B/generate_modify.py --input_file /raid/data/guangwei/copyright_newVersion/scripts/copybench/Llama_3.1_70B/output_with_predictions.json --prompt_file  /raid/data/guangwei/copyright_newVersion/prompts/prompts.literal.format1.json --output_file /raid/data/guangwei/copyright_newVersion/scripts/copybench/Llama_3.1_70B/outputs.literal.prompt1.Llama-3.1-70B.greedy.json --model /raid/data/guangwei/huggingface/hub/models--meta-llama--Llama-3.1-70B/snapshots/349b2ddb53ce8f2849a6c168a81980ab25258dac/ --n_instances 10000

 python /raid/data/guangwei/copyright_newVersion/scripts/eval_literal_copying.py --input /raid/data/guangwei/copyright_newVersion/scripts/copybench/Llama_3.1_70B/outputs.literal.prompt1.Llama-3.1-70B.greedy.json --output /raid/data/guangwei/copyright_newVersion/scripts/copybench/Llama_3.1_70B/scores-literal-copying.literal.prompt1.Llama-3.1-70B.greedy.json



