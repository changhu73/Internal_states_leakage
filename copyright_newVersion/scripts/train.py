import json
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, TensorDataset

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
tokenizer.pad_token = tokenizer.eos_token

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def extract_hidden_states(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_state = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
    return hidden_state

def prepare_training_data(infringement_data, non_infringement_data, test_size=0.2):
    inputs = []
    labels = []

    total_data = len(infringement_data) + len(non_infringement_data)
    with tqdm(total=total_data, desc="Processing data") as pbar:
        for entry in infringement_data:
            hidden_state = extract_hidden_states(entry['input'], model, tokenizer)
            inputs.append(hidden_state)
            labels.append(0)
            pbar.update(1)

        for entry in non_infringement_data:
            hidden_state = extract_hidden_states(entry['input'], model, tokenizer)
            inputs.append(hidden_state)
            labels.append(1)
            pbar.update(1)

    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)

    train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size=test_size, random_state=42)

    return train_inputs, test_inputs, train_labels, test_labels

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

    train_inputs, test_inputs, train_labels, test_labels = prepare_training_data(infringement_data, non_infringement_data)

    input_dim = 768
    hidden_dim = 128
    batch_size = 16 

    model = CustomMLP(input_dim, hidden_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = TensorDataset(train_inputs, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    num_epochs = 100

    for epoch in range(num_epochs):
        start_time = time.time()

        epoch_loss = 0
        epoch_accuracy = 0

        for batch_inputs, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_inputs).squeeze()
            loss = criterion(outputs, batch_labels.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += calculate_accuracy(outputs, batch_labels)

        epoch_loss /= len(train_loader)
        epoch_accuracy /= len(train_loader)

        elapsed_time = time.time() - start_time
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Time: {elapsed_time:.2f}s')

    model.eval()
    with torch.no_grad():
        test_dataset = TensorDataset(test_inputs, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        test_loss = 0
        test_accuracy = 0
        all_outputs = []
        all_labels = []

        for batch_inputs, batch_labels in test_loader:
            outputs = model(batch_inputs).squeeze()
            loss = criterion(outputs, batch_labels.float())
            test_loss += loss.item()
            test_accuracy += calculate_accuracy(outputs, batch_labels)
            all_outputs.append(outputs)
            all_labels.append(batch_labels)

        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader)

        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)

        precision, recall, f1 = calculate_metrics(all_outputs, all_labels)

        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        print(f'Test Precision: {precision:.4f}, Test Recall: {recall:.4f}, Test F1 Score: {f1:.4f}')

    torch.save(model.state_dict(), '/home/Guangwei/sit/copy-bench/models/custom_mlp_model.pt')
    print("模型已保存为 'custom_mlp_model.pt'")

if __name__ == '__main__':
    main()
