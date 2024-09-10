from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
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

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)

tokenizer.pad_token = tokenizer.eos_token

with open('/home/Guangwei/sit/copy-bench/test_division/literal.non_infringement.json', 'r', encoding='utf-8') as file:
    non_infringement_json_data = json.load(file)

# both ok: normal or shuffled
non_infringement_outputs = []
y_non_infringement = []
for entry in non_infringement_json_data:
    non_infringement_outputs.append(entry['input'])
    y_non_infringement.append(1)

with open('/home/Guangwei/sit/copy-bench/test_division/literal.infringement.json', 'r', encoding='utf-8') as file:
    infringement_json_data = json.load(file)

infringement_outputs = []
y_infringement = []
for entry in infringement_json_data:
    infringement_outputs.append(entry['input'])
    y_infringement.append(0)

y_non_infringement = np.array(y_non_infringement)
y_infringement = np.array(y_infringement)


def extract_hidden_states(texts, model, tokenizer, batch_size=4):
    hidden_states = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing data batches"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        hidden_states.append(outputs.hidden_states[-1].mean(dim=1).cpu().numpy())
    return np.vstack(hidden_states)

print("Extracting hidden states for non_infringement texts...")
hidden_states_non_infringement = extract_hidden_states(non_infringement_outputs, model, tokenizer)

print("Extracting hidden states for infringement texts...")
hidden_states_infringement = extract_hidden_states(infringement_outputs, model, tokenizer)

X_non_infringement = hidden_states_non_infringement
X_infringement = hidden_states_infringement

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
custom_mlp = CustomMLP(input_dim, hidden_dim)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(custom_mlp.parameters(), lr=0.001)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

losses = []

for epoch in tqdm(range(500), desc="Training Epochs"):
    custom_mlp.train()
    optimizer.zero_grad()
    outputs = custom_mlp(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{500}, Loss: {loss.item():.4f}")
        
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

print(f"Final Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=["infringement", "non_infringement"]))

# torch.save(custom_mlp.state_dict(), '/home/Guangwei/sit/copy-bench/models/custom_mlp_model.pth')
# print("Model saved to '/home/Guangwei/sit/copy-bench/models/custom_mlp_model.pth'.")

checkpoint = {
    'epoch': epoch + 1,
    'model_state_dict': custom_mlp.state_dict(), 
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
}

torch.save(checkpoint, '/home/Guangwei/sit/copy-bench/models/custom_mlp_checkpoint.ckpt')
print("Checkpoint saved to '/home/Guangwei/sit/copy-bench/models/custom_mlp_checkpoint.ckpt'.")