import json
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge import Rouge
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm

class CustomMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomMLP, self).__init__()
        self.down = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.up = nn.Linear(hidden_dim, 1)
        self.activation = nn.SiLU()

    def forward(self, x):
        down_output = self.down(x)
        gate_output = self.gate(x)
        gated_output = down_output * self.activation(gate_output)
        return self.up(gated_output)

model_name = "EleutherAI/gpt-neox-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to("cuda" if torch.cuda.is_available() else "cpu")

with open("generate/literal_outputs.json", "r") as f:
    data = json.load(f)

input_dim = model.config.hidden_size
hidden_dim = 256
custom_mlp = CustomMLP(input_dim, hidden_dim)

try:
    custom_mlp.load_state_dict(torch.load("models/trained_mlp.pth"))
    print("Load MLP model successfully。")
except FileNotFoundError:
    print("Pretrained model cannot be found. Training a new one...")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(custom_mlp.parameters(), lr=1e-4)
    training_data = [(torch.randn(input_dim), torch.tensor([1.0])) for _ in range(100)]

    custom_mlp.train()
    for epoch in range(10):
        for inputs, labels in training_data:
            optimizer.zero_grad()
            outputs = custom_mlp(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    torch.save(custom_mlp.state_dict(), "models/trained_mlp.pth")
    print("MLP model kept...")

custom_mlp.eval()

rouge = Rouge()
threshold = 0.8

def extract_hidden_state(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.hidden_states[-1].mean(dim=1).cpu()

def get_internal_state(text):
    hidden_state = extract_hidden_state(text)
    with torch.no_grad():
        logits = custom_mlp(hidden_state)
        prob = torch.sigmoid(logits).item()
        return "compliance" if prob > 0.5 else "refuse"

results = []
for entry in tqdm(data, desc="Evaluating outputs"):
    input_text = entry['input']
    generated_output = entry['output']
    
    internal_state = get_internal_state(generated_output)
    if (internal_state == "refuse"):
        label = 0
    else:
        reference_text = entry['preference']
        rouge_score = rouge.get_scores(generated_output, reference_text)[0]['rouge-l']['f']
        label = 0 if rouge_score > threshold else 1
    
    results.append({
        'input': input_text,
        'output': generated_output,
        'label': label
    })

with open("generate/labeled_outputs.json", "w") as f:
    json.dump(results, f, indent=2)

print("Label results saved... 'generate/labeled_outputs.json'.")
