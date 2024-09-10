export PYTHONPATH=./src:$$PYTHONPATH

quick_start:
	$(eval HF_MODEL=meta-llama/Meta-Llama-3.1-8B)
	$(eval MODEL_TAG=Meta-Llama-3.1-8B)
	$(eval N=10)

	### Evaluating literal copying

	/home/Guangwei/.conda/envs/sit/bin/python /home/Guangwei/sit/copy-bench/scripts/generate.py \
		--input_file /home/Guangwei/sit/copy-bench/data/data.literal.json \
		--prompt_file /home/Guangwei/sit/copy-bench/prompts/prompts.literal.format1.json \
		--output_file /home/Guangwei/sit/copy-bench/outputs/outputs.literal.prompt1.$(MODEL_TAG).greedy.json \
		--model $(HF_MODEL) \
		--n_instances 758 \

	/home/Guangwei/.conda/envs/sit/bin/python /home/Guangwei/sit/copy-bench/scripts/eval_literal_copying.py\
		--input /home/Guangwei/sit/copy-bench/outputs/outputs.literal.prompt1.$(MODEL_TAG).greedy.json \
		--output /home/Guangwei/sit/copy-bench/scores/scores-literal-copying.literal.prompt1.$(MODEL_TAG).greedy.json

	# python scripts/eval_quality.py \
	# 	--input outputs/outputs.literal.prompt1.$(MODEL_TAG).greedy.json \
	# 	--output scores/scores-quality.literal.prompt1.$(MODEL_TAG).greedy.json

	#### Evaluate non-literal copying
	/home/Guangwei/.conda/envs/sit/bin/python /home/Guangwei/sit/copy-bench/scripts/generate.py \
		--input_file /home/Guangwei/sit/copy-bench/data/data.nonliteral.json \
		--prompt_file /home/Guangwei/sit/copy-bench/prompts/prompts.nonliteral.format1.json \
		--output_file outputs/outputs.nonliteral.prompt1.$(MODEL_TAG).greedy.json \
		--model $(HF_MODEL) \
		--max_new_tokens 1024 \
		--n_instances 1000 \

	/home/Guangwei/.conda/envs/sit/bin/python /home/Guangwei/sit/copy-bench/scripts/eval_char_copying.py \
		--input outputs/outputs.nonliteral.prompt1.$(MODEL_TAG).greedy.json \
		--output /home/Guangwei/sit/copy-bench/scores/scores-char-copying.nonliteral.prompt1.$(MODEL_TAG).greedy.json

	/home/Guangwei/.conda/envs/sit/bin/python /home/Guangwei/sit/copy-bench/scripts/eval_event_copying.py \
		--input outputs/outputs.nonliteral.prompt1.$(MODEL_TAG).greedy.json \
		--output /home/Guangwei/sit/copy-bench/scores/scores-event-copying.nonliteral.prompt1.$(MODEL_TAG).greedy.json

	# python scripts/eval_quality.py \
	# 	--input outputs/outputs.nonliteral.prompt1.$(MODEL_TAG).greedy.json \
	# 	--output scores/scores-quality.nonliteral.prompt1.$(MODEL_TAG).greedy.json

	#### Evaluate Fact Recall
	/home/Guangwei/.conda/envs/sit/bin/python /home/Guangwei/sit/copy-bench/scripts/generate.py \
		--input_file /home/Guangwei/sit/copy-bench/data/data.qa.json \
		--prompt_file  /home/Guangwei/sit/copy-bench/prompts/prompts.qa.format1.json \
		--output_file outputs/outputs.qa.prompt1.Llama-2-7b-hf.greedy.json \
		--model meta-llama/Llama-2-7b-hf \
		--n_instances 589 \

	python scripts/eval_qa.py \
		--input outputs/outputs.qa.prompt1.$(MODEL_TAG).greedy.json \
		--output scores/scores-qa.qa.prompt1.$(MODEL_TAG).greedy.json

	python scripts/summary.py --root ./scores/**.json
