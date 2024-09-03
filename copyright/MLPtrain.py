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

model_name = "EleutherAI/gpt-neox-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)

tokenizer.pad_token = tokenizer.eos_token

# with open('/home/Guangwei/sit/copy-bench/label/literal_output.json', 'r', encoding='utf-8') as file:
#     compliance_json_data = json.load(file)
# compliance_outputs = [entry['output'] for entry in compliance_json_data]
# y_compliance = [entry['label'] for entry in compliance_json_data]

# with open('/home/Guangwei/sit/copy-bench/label/literal_reference.json', 'r', encoding='utf-8') as file:
#     refuse_json_data = json.load(file)
# refuse_outputs = [entry['reference'] for entry in refuse_json_data]
# y_refuse = [entry['label'] for entry in refuse_json_data]

with open('/home/Guangwei/sit/copy-bench/label/literal_output.json', 'r', encoding='utf-8') as file:
    compliance_json_data = json.load(file)

# both ok: normal or shuffled
compliance_outputs = []
y_compliance = []
for entry in compliance_json_data:
    if 'output' in entry and entry['output'] is not None:
        compliance_outputs.append(entry['output'])
    elif 'reference' in entry and entry['reference'] is not None:
        compliance_outputs.append(entry['reference'])
    y_compliance.append(entry.get('label', 2))

with open('/home/Guangwei/sit/copy-bench/label/literal_reference.json', 'r', encoding='utf-8') as file:
    refuse_json_data = json.load(file)

refuse_outputs = []
y_refuse = []
for entry in refuse_json_data:
    if 'reference' in entry and entry['reference'] is not None:
        refuse_outputs.append(entry['reference'])
    elif 'output' in entry and entry['output'] is not None:
        refuse_outputs.append(entry['output'])
    y_refuse.append(entry.get('label', 2))

y_compliance = np.array(y_compliance)
y_refuse = np.array(y_refuse)


def extract_hidden_states(texts, model, tokenizer, batch_size=4):
    hidden_states = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing data batches"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        hidden_states.append(outputs.hidden_states[-1].mean(dim=1).cpu().numpy())
    return np.vstack(hidden_states)

print("Extracting hidden states for compliance texts...")
hidden_states_compliance = extract_hidden_states(compliance_outputs, model, tokenizer)

print("Extracting hidden states for refuse texts...")
hidden_states_refuse = extract_hidden_states(refuse_outputs, model, tokenizer)

X_compliance = hidden_states_compliance
X_refuse = hidden_states_refuse

split_index_compliance = int(0.8 * len(X_compliance))
X_compliance_train = X_compliance[:split_index_compliance]
X_compliance_test = X_compliance[split_index_compliance:]
y_compliance_train = y_compliance[:split_index_compliance]
y_compliance_test = y_compliance[split_index_compliance:]

split_index_refuse = int(0.8 * len(X_refuse))
X_refuse_train = X_refuse[:split_index_refuse]
X_refuse_test = X_refuse[split_index_refuse:]
y_refuse_train = y_refuse[:split_index_refuse]
y_refuse_test = y_refuse[split_index_refuse:]

X_train = np.vstack((X_compliance_train, X_refuse_train))
X_test = np.vstack((X_compliance_test, X_refuse_test))
y_train = np.concatenate((y_compliance_train, y_refuse_train))
y_test = np.concatenate((y_compliance_test, y_refuse_test))

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
print(classification_report(y_test, y_pred, target_names=["Refuse", "Compliance"]))

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
