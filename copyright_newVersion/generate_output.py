import sys, os
import json
from tqdm import tqdm
import random
random.seed(42)

def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f'[INFO] Save results to {file_path}')

def apply_transform(apply_config, instance):
    instance = instance.copy()
    if "input" in  apply_config:
        assert apply_config["input"] == "capitalize"
        if "input" in instance:
            instance["input"] = instance["input"].upper()
    if "output" in  apply_config:
        assert apply_config["output"] == "capitalize"
        if "output" in instance:
            instance["output"] = instance["output"].upper()
    return instance

def process_conversation(prompt_config, instance, shots=0):
    # sample shots from the instance
    import random, string, re
    demos = random.sample(prompt_config["demos"], shots) if shots > 0 else []
    demo_prompt_template = string.Template(prompt_config["demo_prompt"])
    instruction = prompt_config["instruction"]
    demo_sep = prompt_config["demo_sep"]
    apply_config = prompt_config["apply_config"] if "apply_config" in prompt_config else {}
    demo_prompt_list = [demo_prompt_template.safe_substitute(**apply_transform(apply_config, demo)) for demo in demos]
    prompt = instruction + demo_sep + demo_sep.join(demo_prompt_list)
    if "task_instruction" in prompt_config:
        prompt = prompt + demo_sep + prompt_config["task_instruction"]
    if "task_prompt" in prompt_config:
        task_prompt_template = string.Template(prompt_config["task_prompt"])
        prompt = prompt + demo_sep + task_prompt_template.safe_substitute(**apply_transform(apply_config, instance), output="")
    else:
        prompt = prompt + demo_sep + demo_prompt_template.safe_substitute(**apply_transform(apply_config, instance), output="")
    # assert no pattern ${...} left in the prompt
    assert re.search(r"\${.*?}", prompt) is None, f"Unresolved pattern in prompt: {prompt}"
    prompt = prompt.strip()
    if prompt_config.get("enable_system", False):
        return [{"role": "system", "content": prompt_config["system"]}, {"role": "user", "content": prompt}]
    return [{"role": "user", "content": prompt}]

# if the model do not support system prompt, combine system prompt with the user instruction
def merge_conversation(conversation):
    # assert len(conversation) <= 2, f"Invalid conversation length: {len(conversation)}"
    # if conversation[0]["role"] == "system":
    #     assert conversation[1]["role"] == "user", f"Invalid conversation roles: {conversation[0]['role']} -> {conversation[1]['role']}"
    #     prompt = conversation[0]["content"] + "\n\n" + conversation[1]["content"]
    #     return [{"role": "user", "content": prompt}]
    # return conversation
    return "\n\n".join([msg["content"] for msg in conversation])


def main(args):
    with open(args.input_file, "r") as f:
        snippets = json.load(f)
    if args.n_instances is not None:
        # snippets = snippets[:args.n_instances]
        # random sampling
        if args.n_instances < len(snippets):
            snippets = random.sample(snippets, args.n_instances)
        print(f"[INFO] Randomly sampled {args.n_instances} instances from {len(snippets)} instances.")

    # process prompts
    if args.prompt_file is None:
        prompt_config = None
    else:
        with open(args.prompt_file, "r") as f:
            prompt_config = json.load(f)
    assert prompt_config is not None, f"Prompt config is not provided."
    prompt_tag = prompt_config.get("tag", None)

    if args.system_prompt:
        prompt_config["enable_system"] = True
    for s in snippets:
        s["conversation"] = process_conversation(prompt_config, s, shots=args.shots)

    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            output_list = json.load(f)
        print(f"[INFO] {len(output_list)} samples already exist in {args.output_file}")
        # remove snippets that are already processed
        existing_ids = [s["id"] for s in output_list]
        snippets = [s for s in snippets if s["id"] not in existing_ids]
    else:
        output_list = []

    if "llama" in args.model.lower():  # Llama model case
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        # Load Llama model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        # Set the model to run on GPU or CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print(f"[INFO] Model loaded: {args.model}")

        # Test the model with a simple prompt
        test_input = "how are you?"
        inputs = tokenizer(test_input, return_tensors="pt").to(device)
        with torch.no_grad():
            test_outputs = model.generate(**inputs, max_new_tokens=50, temperature=1)
        test_output_text = tokenizer.decode(test_outputs[0], skip_special_tokens=True)
        print(f"[INFO] Test output: {test_output_text}")

        # Process the actual snippets
        with tqdm(total=len(snippets)) as pbar:
            for i in range(0, len(snippets), args.batch_size):
                batch_snippets = snippets[i:i + args.batch_size]
                prompt_list = [merge_conversation(s["conversation"]) for s in batch_snippets]

                tokenizer.pad_token = tokenizer.eos_token
                inputs = tokenizer(prompt_list, return_tensors="pt", padding=True, truncation=True, max_length = 512).to(device)

                # Generate outputs from the model
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=args.max_new_tokens,
                        temperature= None,
                        do_sample=False, 
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty
                    )

                # Decode outputs
                output_text_list = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

                # Append results to output list
                for s, prompt, output_text in zip(batch_snippets, prompt_list, output_text_list):
                    output_list.append({
                        "model": args.model,
                        "prompt_tag": prompt_tag,
                        "shots": args.shots,
                        "decoding": args.decoding,
                        **s,
                        "prompt": prompt,
                        "output": output_text,
                    })
                pbar.update(args.batch_size)

                # Save intermediate results
                save_json(output_list, args.output_file)

        # Save final results
        save_json(output_list, args.output_file)
    
    print("[INFO] Exiting...")
    exit(0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n_instances", type=int, default=None)

    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--format", choices=["default", "chat", "context"], default="default")
    # batch_size
    parser.add_argument("--batch_size", type=int, default=8)
    # decoding parameters
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    # repetition_penalty
    parser.add_argument("--repetition_penalty", type=float, default=1.1)

    parser.add_argument("--memfree", action="store_true")
    parser.add_argument("--memfree_rejection_file", type=str, default=None)
    parser.add_argument("--memfree_n", type=int, default=10)

    parser.add_argument("--system_prompt", action="store_true")

    # parser.add_argument("--copyright", action="store_true")
    # parser.add_argument("--copyright_alpha", type=float, default=0.5)

    args = parser.parse_args()
    if args.memfree and not args.system_prompt and args.temperature == 0.0:
        args.decoding = "memfree"
    elif args.system_prompt and not args.memfree and args.temperature == 0.0:
        args.decoding = "system"
    elif args.temperature == 0.0 and not args.memfree and not args.system_prompt:
        args.decoding = "greedy"
    elif args.temperature > 0.0 and not args.memfree and not args.system_prompt:
        args.decoding = "sampling"
    else:
        args.decoding = "undefined"
        print(f'[WARNING] Unsupported decoding mode: {args}')
    main(args)
