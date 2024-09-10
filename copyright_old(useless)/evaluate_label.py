import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge import Rouge
from tqdm import tqdm
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# model_name = "EleutherAI/gpt-neox-20b"
# tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
# tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to("cuda" if torch.cuda.is_available() else "cpu")

with open("/home/Guangwei/sit/copy-bench/cleaned/literal_data.json", "r") as f:
    data = json.load(f)

rouge = Rouge()
threshold = 0.8

results = []
for entry in tqdm(data, desc="Evaluating outputs"):
    input_text = entry['input']
    generated_output = entry['output']
    reference_text = entry['reference']
    
    rouge_score = rouge.get_scores(generated_output, reference_text)[0]['rouge-l']['f']
    print(rouge_score)
    label = 0 if rouge_score > threshold else 1
    
    results.append({
        'id': entry['id'],
        'title': entry['title'],
        'input': input_text,
        'output': generated_output,
        'reference': reference_text,
        'label': label
    })

folder_path = "/home/Guangwei/sit/copy-bench/label/"
file_name = 'literal_labels.json'
os.makedirs(folder_path, exist_ok=True)
file_path = os.path.join(folder_path, file_name)
with open(file_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved to '{file_path}'.")
