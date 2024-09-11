import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

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

def extract_hidden_states(texts, model, tokenizer, batch_size=4):
    hidden_states = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing data batches"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        hidden_states.append(outputs.hidden_states[-1].mean(dim=1).cpu().numpy())
    return np.vstack(hidden_states)

def load_data(non_infringement_file, infringement_file):
    with open(non_infringement_file, 'r', encoding='utf-8') as file:
        non_infringement_json_data = json.load(file)

    non_infringement_outputs = [entry['input'] for entry in non_infringement_json_data]
    y_non_infringement = [1] * len(non_infringement_outputs)

    with open(infringement_file, 'r', encoding='utf-8') as file:
        infringement_json_data = json.load(file)

    infringement_outputs = [entry['input'] for entry in infringement_json_data]
    y_infringement = [0] * len(infringement_outputs)

    return non_infringement_outputs, y_non_infringement, infringement_outputs, y_infringement

def train_model(X_train, y_train, X_test, y_test, input_dim, hidden_dim, epochs=200, lr=0.001):
    custom_mlp = CustomMLP(input_dim, hidden_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(custom_mlp.parameters(), lr=lr)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    losses = []
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        custom_mlp.train()
        optimizer.zero_grad()
        outputs = custom_mlp(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
            
            custom_mlp.eval()
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            with torch.no_grad():
                y_pred_logits = custom_mlp(X_test_tensor)
                y_pred = (torch.sigmoid(y_pred_logits) > 0.5).float().numpy()
            
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Test Accuracy at Epoch {epoch + 1}: {accuracy * 100:.2f}%")

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

    return custom_mlp, losses, accuracy

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to '{filepath}'.")

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=512)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, output_hidden_states=True)
    tokenizer.pad_token = tokenizer.eos_token

    non_infringement_outputs, y_non_infringement, infringement_outputs, y_infringement = load_data(
        args.non_infringement_file, args.infringement_file
    )
    
    y_non_infringement = np.array(y_non_infringement)
    y_infringement = np.array(y_infringement)

    print("Extracting hidden states for non_infringement texts...")
    X_non_infringement = extract_hidden_states(non_infringement_outputs, model, tokenizer)

    print("Extracting hidden states for infringement texts...")
    X_infringement = extract_hidden_states(infringement_outputs, model, tokenizer)

    split_index_non_infringement = int(0.8 * len(X_non_infringement))
    X_non_infringement_train = X_non_infringement[:split_index_non_infringement]
    X_non_infringement_test = X_non_infringement[split_index_non_infringement:]
    y_non_infringement_train = y_non_infringement[:split_index_non_infringement]
    y_non_infringement_test = y_non_infringement[split_index_non_infringement:]

    split_index_infringement = int(0.8 * len(X_infringement))
    X_infringement_train = X_infringement[:split_index_infringement]
    X_infringement_test = X_infringement[split_index_infringement:]
    y_infringement_train = y_infringement[:split_index_infringement]
    y_infringement_test = y_infringement[split_index_infringement:]

    X_train = np.vstack((X_non_infringement_train, X_infringement_train))
    X_test = np.vstack((X_non_infringement_test, X_infringement_test))
    y_train = np.concatenate((y_non_infringement_train, y_infringement_train))
    y_test = np.concatenate((y_non_infringement_test, y_infringement_test))

    print("Data successfully split into training and test sets.")

    input_dim = X_train.shape[1]
    hidden_dim = 256 

    custom_mlp, losses, accuracy = train_model(X_train, y_train, X_test, y_test, input_dim, hidden_dim)

    save_checkpoint(custom_mlp, torch.optim.Adam(custom_mlp.parameters()), len(losses), losses[-1], args.checkpoint_file)

    print(f"Final Model Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, (torch.sigmoid(torch.tensor(custom_mlp(torch.tensor(X_test, dtype=torch.float32)))) > 0.5).float().numpy(), target_names=["infringement", "non_infringement"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Custom MLP for infringement detection.")
    parser.add_argument('--model_name', type=str, help='Name of the pretrained model.', default='meta-llama/Meta-Llama-3.1-8B')
    parser.add_argument('--non_infringement_file', type=str, help='Path to the non-infringement data file.', default='/home/guangwei/LLM-COPYRIGHT/copyright_newVersion/test_division/literal.non_infringement.json')
    parser.add_argument('--infringement_file', type=str, help='Path to the infringement data file.', default='/home/guangwei/LLM-COPYRIGHT/copyright_newVersion/test_division/literal.infringement.json')
    parser.add_argument('--checkpoint_file', type=str, help='Path to save the model checkpoint.', default='/home/guangwei/LLM-COPYRIGHT/copyright_newVersion/models/custom_mlp_model.pth')

    args = parser.parse_args()
    main(args)
