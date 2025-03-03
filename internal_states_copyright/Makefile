/bin/python3 /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scripts/generate.py --input_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/data/data.literal.json --prompt_file  /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/prompts/prompts.literal.format1.json --output_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/outputs/outputs.literal.prompt1.Meta-Llama-3.1-8B.greedy.json --model meta-llama/Meta-Llama-3.1-8B --n_instances 758
/bin/python3 /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scripts/generate.py --input_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/data/data.nonliteral.json --prompt_file  /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/prompts/prompts.nonliteral.format1.json --output_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/outputs/outputs.nonliteral.prompt1.Meta-Llama-3.1-8B.greedy.json --model meta-llama/Meta-Llama-3.1-8B --n_instances 590
/bin/python3 /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scripts/generate.py --input_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/data/data.qa.json --prompt_file  /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/prompts/prompts.qa.format1.json --output_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/outputs/outputs.qa.prompt1.Meta-Llama-3.1-8B.greedy.json --model meta-llama/Meta-Llama-3.1-8B --n_instances 589



evaluate:
/home/Guangwei/.conda/envs/sit/bin/python /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scripts/eval_literal_copying.py --input /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/outputs/outputs.literal.prompt1.Meta-Llama-3.1-8B.greedy.json --output /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scores/scores-literal-copying.literal.prompt1.Meta-Llama-3.1-8B.greedy.json


/home/Guangwei/.conda/envs/sit/bin/python /home/Guangwei/copy-bench/scripts/generate.py --input_file /home/Guangwei/copy-bench/data/data.literal.json --prompt_file  /home/Guangwei/copy-bench/prompts/prompts.literal.format2.json --output_file /home/Guangwei/copy-bench/outputs/outputs.literal.prompt2.Meta-Llama-3.1-8B.greedy.json --model meta-llama/Meta-Llama-3.1-8B --n_instances 758



eval:
/home/guangwei/miniconda3/envs/zdh/bin/python /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scripts/eval_literal_copying.py --input /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/outputs/outputs.literal.prompt1.Llama-3.1-70B.greedy.json --output /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scores/scores-literal-copying.literal.prompt1.Llama-3.1-70B.greedy.json


export PYTHONPATH=./src:$$PYTHONPATH

meta-llama/Llama-2-70b-chat-hf

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

	/bin/python3 /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scripts/generate.py --input_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/data/data.literal.json --prompt_file  /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/prompts/prompts.literal.format1.json --output_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/outputs/outputs.literal.prompt1.Meta-Llama-3.1-8B.greedy.json --model meta-llama/Meta-Llama-3.1-8B --n_instances 758

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

	/bin/python3 /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scripts/generate.py --input_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/data/data.nonliteral.json --prompt_file  /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/prompts/prompts.nonliteral.format1.json --output_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/outputs/outputs.nonliteral.prompt1.Meta-Llama-3.1-8B.greedy.json --model meta-llama/Meta-Llama-3.1-8B --n_instances 1000

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
	/home/Guangwei/.conda/envs/sit/bin/python /home/Guangwei/copy-bench/scripts/generate.py \
		--input_file /home/Guangwei/copy-bench/data/data.qa.json \
		--prompt_file  /home/Guangwei/copy-bench/prompts/prompts.qa.format1.json \
		--output_file outputs/outputs.qa.prompt1.Meta-Llama-3.1-8B.greedy.json \
		--model meta-llama/Meta-Llama-3.1-8B \
		--n_instances 589 \

	/home/guangwei/miniconda3/envs/zdh/bin/python /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scripts/generate.py 
	--input_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/data/data.qa.json 
	--prompt_file  /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/prompts/prompts.qa.format1.json --output_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/outputs/outputs.qa.prompt1.Meta-Llama-3.1-8B.greedy.json --model meta-llama/Meta-Llama-3.1-8B --n_instances 10000

	python scripts/eval_qa.py \
		--input outputs/outputs.qa.prompt1.$(MODEL_TAG).greedy.json \
		--output scores/scores-qa.qa.prompt1.$(MODEL_TAG).greedy.json

eval_qa:
/home/guangwei/miniconda3/envs/zdh/bin/python /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scripts/eval_qa.py --input /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/outputs/outputs.qa.prompt1.Llama-3.1-70B.greedy.json --output /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scores/scores-qa.qa.prompt1.Llama-3.1-70B.greedy.json



	python scripts/summary.py --root ./scores/**.json

generate_literal:
/home/guangwei/miniconda3/envs/zdh/bin/python /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scripts/generate.py --input_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/data/data.extra.json --prompt_file  /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/prompts/prompts.literal.format1.json --output_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/outputs/outputs.literal.prompt1.Llama-3.1-8B.greedy.json --model meta-llama/Meta-Llama-3.1-8B --n_instances 10000
/home/guangwei/miniconda3/envs/zdh/bin/python /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scripts/generate.py --input_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/data/data.qa.json --prompt_file  /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/prompts/prompts.qa.format1.json --output_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/outputs/outputs.qa.prompt1.Llama-3.1-70B.greedy.json --model meta-llama/Llama-3.1-70B --n_instances 10000
/home/guangwei/miniconda3/envs/zdh/bin/python /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scripts/generate.py --input_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/data/data.nonliteral.json --prompt_file  /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/prompts/prompts.nonliteral.format1.json --output_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/outputs/outputs.nonliteral.prompt1.Llama-3.1-70B.greedy.json --model meta-llama/Llama-3.1-70B --n_instances 10000


non:
/home/guangwei/miniconda3/envs/zdh/bin/python /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scripts/generate.py --input_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/data/data.nonliteral.json --prompt_file  /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/prompts/prompts.nonliteral.format1.json --output_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/outputs/outputs.nonliteral.prompt1.Meta-Llama-3.1-8B.greedy.json --model meta-llama/Meta-Llama-3.1-8B --n_instances 10000

/home/guangwei/miniconda3/envs/zdh/bin/python /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scripts/eval_char_copying.py --input /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/outputs/outputs.nonliteral.prompt1.Meta-Llama-3.1-8B.greedy.json --output /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scores/scores-char-copying.nonliteral.prompt1.Meta-Llama-3.1-8B.greedy.json

/home/guangwei/miniconda3/envs/zdh/bin/python /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scripts/eval_event_copying.py --input /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/outputs/outputs.nonliteral.prompt1.Meta-Llama-3.1-8B.greedy.json --output /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scores/scores-event-copying.nonliteral.prompt1.Meta-Llama-3.1-8B.greedy.json







/home/guangwei/miniconda3/envs/zdh/bin/python /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scripts/generate.py --input_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/data/data.literal.json --prompt_file  /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/prompts/prompts.literal.format1.json --output_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/new_outputs/outputs.literal.prompt1.Llama-3.1-70B.greedy.json --model /raid/data/guangwei/huggingface/hub/models--meta-llama--Llama-3.1-70B/snapshots/349b2ddb53ce8f2849a6c168a81980ab25258dac/ --n_instances 10000

/home/guangwei/miniconda3/envs/zdh/bin/python /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scripts/eval_literal_copying.py --input /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/new_outputs/outputs.literal.prompt1.Llama-3.1-70B.greedy.json --output /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/new_scores/scores-literal-copying.literal.prompt1.Llama-3.1-70B.greedy.json


/home/guangwei/miniconda3/envs/zdh/bin/python /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scripts/generate.py --input_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/data/data.nonliteral.json --prompt_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/prompts/prompts.nonliteral.format1.json --output_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/new_outputs/outputs.nonliteral.prompt1.Llama-3.1-70B.greedy.json --model /raid/data/guangwei/huggingface/hub/models--meta-llama--Llama-3.1-70B/snapshots/349b2ddb53ce8f2849a6c168a81980ab25258dac/ --max_new_tokens 1024

/home/guangwei/miniconda3/envs/zdh/bin/python /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scripts/eval_char_copying.py --input /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/new_outputs/outputs.nonliteral.prompt1.Llama-3.1-70B.greedy.json --output /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/new_scores/scores-char-copying.nonliteral.prompt1.Llama-3.1-70B.greedy.json

/home/guangwei/miniconda3/envs/zdh/bin/python /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scripts/eval_event_copying.py --input /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/new_outputs/outputs.nonliteral.prompt1.Llama-3.1-70B.greedy.json --output /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/new_scores/scores-event-copying.nonliteral.prompt1.Llama-3.1-70B.greedy.json


/home/guangwei/miniconda3/envs/zdh/bin/python /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scripts/generate.py --input_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/data/data.qa.json --prompt_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/prompts/prompts.qa.format1.json --output_file /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/new_outputs/outputs.qa.prompt1.Llama-3.1-70B.greedy.json --model /raid/data/guangwei/huggingface/hub/models--meta-llama--Llama-3.1-70B/snapshots/349b2ddb53ce8f2849a6c168a81980ab25258dac/

/home/guangwei/miniconda3/envs/zdh/bin/python /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scripts/eval_qa.py --input /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/new_outputs/outputs.qa.prompt1.Llama-3.1-70B.greedy.json --output /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/new_scores/scores-qa.qa.prompt1.Llama-3.1-70B.greedy.json

/home/guangwei/miniconda3/envs/zdh/bin/python /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scripts/eval_fluency.py --input /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/new_outputs/outputs.literal.prompt1.Llama-3.1-70B.greedy.json --output /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/new_scores/scores-fluency.literal.prompt1.Llama-3.1-70B.greedy.json

/home/guangwei/miniconda3/envs/zdh/bin/python /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scripts/eval_fluency.py --input /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/new_outputs/outputs.nonliteral.prompt1.Llama-3.1-70B.greedy.json --output /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/new_scores/scores-fluency.nonliteral.prompt1.Llama-3.1-70B.greedy.json


/home/guangwei/miniconda3/envs/zdh/bin/python /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scripts/summary.py --root /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/new_scores/scores-literal-copying.literal.prompt1.Llama-3.1-70B.greedy.json /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/new_scores/scores-char-copying.nonliteral.prompt1.Llama-3.1-70B.greedy.json /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/new_scores/scores-event-copying.nonliteral.prompt1.Llama-3.1-70B.greedy.json /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/new_scores/scores-qa.qa.prompt1.Llama-3.1-70B.greedy.json /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/new_scores/scores-fluency.literal.prompt1.Llama-3.1-70B.greedy.json /home/guangwei/LLM-COPYRIGHT/copyright_newVersion/new_scores/scores-fluency.nonliteral.prompt1.Llama-3.1-70B.greedy.json









new raid:

/home/guangwei/miniconda3/envs/zdh/bin/python /raid/data/guangwei/copyright_newVersion/scripts/copybench/Llama_3.1_70B/generate_modify.py --input_file /raid/data/guangwei/copyright_newVersion/scripts/copybench/Llama_3.1_70B/output_with_predictions.json --prompt_file  /raid/data/guangwei/copyright_newVersion/prompts/prompts.literal.format1.json --output_file /raid/data/guangwei/copyright_newVersion/scripts/copybench/Llama_3.1_70B/outputs.literal.prompt1.Llama-3.1-70B.greedy.json --model /raid/data/guangwei/huggingface/hub/models--meta-llama--Llama-3.1-70B/snapshots/349b2ddb53ce8f2849a6c168a81980ab25258dac/ --n_instances 10000

/home/guangwei/miniconda3/envs/zdh/bin/python /raid/data/guangwei/copyright_newVersion/scripts/eval_literal_copying.py --input /raid/data/guangwei/copyright_newVersion/scripts/copybench/Llama_3.1_70B/outputs.literal.prompt1.Llama-3.1-70B.greedy.json --output /raid/data/guangwei/copyright_newVersion/scripts/copybench/Llama_3.1_70B/scores-literal-copying.literal.prompt1.Llama-3.1-70B.greedy.json



