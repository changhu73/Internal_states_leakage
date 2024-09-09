import json
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
tokenizer.pad_token = tokenizer.eos_token

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def extract_hidden_states(texts, model, tokenizer, batch_size=4):
    hidden_states = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing data batches"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        hidden_states.append(outputs.hidden_states[-1].mean(dim=1).cpu().numpy())
    return np.vstack(hidden_states)

def prepare_training_data(infringement_data, non_infringement_data):
    inputs = []
    labels = []

    for entry in infringement_data:
        hidden_state = extract_hidden_states(entry['input'], model, tokenizer)
        inputs.append(hidden_state)
        labels.append(0)

    for entry in non_infringement_data:
        hidden_state = extract_hidden_states(entry['input'], model, tokenizer)
        inputs.append(hidden_state)
        labels.append(1)

    return torch.stack(inputs), torch.tensor(labels)

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

def calculate_accuracy(outputs, labels):
    predicted = torch.round(torch.sigmoid(outputs.detach())).cpu().numpy()
    correct_predictions = (predicted == labels.cpu().numpy()).sum()
    accuracy = correct_predictions / labels.size(0)
    return accuracy

def calculate_metrics(outputs, labels):
    predicted = torch.round(torch.sigmoid(outputs.detach())).cpu().numpy()
    labels = labels.cpu().numpy()
    
    precision = precision_score(labels, predicted)
    recall = recall_score(labels, predicted)
    f1 = f1_score(labels, predicted)

    return precision, recall, f1

def main():
    infringement_data = load_data('/home/Guangwei/sit/copy-bench/test_division/literal.infringement.json')
    non_infringement_data = load_data('/home/Guangwei/sit/copy-bench/test_division/literal.non_infringement.json')

    inputs, labels = prepare_training_data(infringement_data, non_infringement_data)

    input_dim = 768
    hidden_dim = 128

    model = CustomMLP(input_dim, hidden_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    num_epochs = 100

    for epoch in range(num_epochs):
        start_time = time.time()
        
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        accuracy = calculate_accuracy(outputs, labels)

        elapsed_time = time.time() - start_time
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Time: {elapsed_time:.2f}s')

    # Evaluate on the test set
    model.eval()
    with torch.no_grad():
        outputs = model(inputs).squeeze()
        test_loss = criterion(outputs, labels.float())
        test_accuracy = calculate_accuracy(outputs, labels)
        
        # Calculate precision, recall, and F1 score
        precision, recall, f1 = calculate_metrics(outputs, labels)
        
        print(f'Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_accuracy:.4f}')
        print(f'Test Precision: {precision:.4f}, Test Recall: {recall:.4f}, Test F1 Score: {f1:.4f}')

    # 保存模型
    torch.save(model.state_dict(), '/home/Guangwei/sit/copy-bench/models/custom_mlp_model.pt')
    print("模型已保存为 'custom_mlp_model.pt'")

if __name__ == '__main__':
    main()
