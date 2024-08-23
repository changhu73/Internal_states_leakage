from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Define the custom MLP architecture
class CustomMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomMLP, self).__init__()
        self.down = nn.Linear(input_dim, hidden_dim)  # Down-projection layer
        self.gate = nn.Linear(input_dim, hidden_dim)  # Gate mechanism
        self.up = nn.Linear(hidden_dim, 1)            # Up-projection layer
        self.activation = nn.SiLU()                   # SiLU activation

    def forward(self, x):
        down_output = self.down(x)
        gate_output = self.gate(x)
        gated_output = down_output * self.activation(gate_output)  # Element-wise multiplication
        return self.up(gated_output)

model_name = "EleutherAI/gpt-neox-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)

tokenizer.pad_token = tokenizer.eos_token

# Load datasets
seen_df = pd.read_csv("dataset/seen_news.csv")
unseen_df = pd.read_csv("dataset/unseen_news.csv")

seen_texts = seen_df['content'].tolist()
unseen_texts = unseen_df['content'].tolist()

def extract_hidden_states(texts, model, tokenizer):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
    return hidden_states

hidden_states_seen = extract_hidden_states(seen_texts, model, tokenizer)
hidden_states_unseen = extract_hidden_states(unseen_texts, model, tokenizer)

X_seen = hidden_states_seen
X_unseen = hidden_states_unseen

y_seen = np.zeros(len(X_seen))
y_unseen = np.ones(len(X_unseen))

X = np.vstack((X_seen, X_unseen))
y = np.concatenate((y_seen, y_unseen))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

# Initialize the custom MLP model
input_dim = X.shape[1]  # Number of features
hidden_dim = 256  # Set hidden layer size
custom_mlp = CustomMLP(input_dim, hidden_dim)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.Adam(custom_mlp.parameters(), lr=0.001)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Add a dimension for binary classification

# Training the model
for epoch in range(200):
    custom_mlp.train()  # Set model to training mode
    optimizer.zero_grad()  # Zero the gradients
    outputs = custom_mlp(X_train_tensor)  # Forward pass
    loss = criterion(outputs, y_train_tensor)  # Compute the loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights

# Evaluate the model
custom_mlp.eval()  # Set model to evaluation mode
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    y_pred_logits = custom_mlp(X_test_tensor)  # Get logits
    y_pred = (torch.sigmoid(y_pred_logits) > 0.5).float().numpy()  # Convert to binary predictions

accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=["seen", "unseen"]))

# # PCA for visualization
# pca = PCA(n_components=2)
# reduced_features = pca.fit_transform(X_test)

# plt.figure(figsize=(10, 6))
# plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=y_test, cmap='viridis')
# plt.title('PCA降维后的特征分布')
# plt.xlabel('1')
# plt.ylabel('2')
# plt.colorbar()
# plt.savefig('test/pca_feature_distribution.png')
