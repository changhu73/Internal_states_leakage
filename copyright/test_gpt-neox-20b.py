from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm 

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
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)

tokenizer.pad_token = tokenizer.eos_token

# Load datasets
print("Loading datasets...")
seen_df = pd.read_csv("dataset/seen_news.csv")
unseen_df = pd.read_csv("dataset/unseen_news.csv")
print("Datasets loaded.")

seen_texts = seen_df['content'].tolist()
unseen_texts = unseen_df['content'].tolist()

# Monitor data processing with tqdm
def extract_hidden_states(texts, model, tokenizer, batch_size=16):
    hidden_states = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing data batches"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        hidden_states.append(outputs.hidden_states[-1].mean(dim=1).cpu().numpy())
    return np.vstack(hidden_states)

print("Extracting hidden states for seen texts...")
hidden_states_seen = extract_hidden_states(seen_texts, model, tokenizer)
print("Extracting hidden states for unseen texts...")
hidden_states_unseen = extract_hidden_states(unseen_texts, model, tokenizer)

X_seen = hidden_states_seen
X_unseen = hidden_states_unseen

y_seen = np.zeros(len(X_seen))
y_unseen = np.ones(len(X_unseen))

split_index_seen = int(0.8 * len(X_seen))
X_seen_train = X_seen[:split_index_seen]
X_seen_test = X_seen[split_index_seen:]
y_seen_train = y_seen[:split_index_seen]
y_seen_test = y_seen[split_index_seen:]

split_index_unseen = int(0.8 * len(X_unseen))
X_unseen_train = X_unseen[:split_index_unseen]
X_unseen_test = X_unseen[split_index_unseen:]
y_unseen_train = y_unseen[:split_index_unseen]
y_unseen_test = y_unseen[split_index_unseen:]

X_train = np.vstack((X_seen_train, X_unseen_train))
X_test = np.vstack((X_seen_test, X_unseen_test))
y_train = np.concatenate((y_seen_train, y_unseen_train))
y_test = np.concatenate((y_seen_test, y_unseen_test))

print("Splitting data into training and test sets...")

X = np.vstack((X_seen, X_unseen))
y = np.concatenate((y_seen, y_unseen))

# # print("Splitting data into training and test sets...")
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# split_index = int(0.8 * len(X))

# X_train = X[:split_index] 
# X_test = X[split_index:] 
# y_train = y[:split_index] 
# y_test = y[split_index:]  
# print("Splitting data into training and test sets...")

# print(f"Training data size: {len(X_train)}")
# print(f"Test data size: {len(X_test)}")
# print(f"Training labels size: {len(y_train)}")
# print(f"Test labels size: {len(y_test)}")

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

# Track loss values for visualization
losses = []

# Training the model with progress monitoring
for epoch in tqdm(range(200), desc="Training Epochs"):
    custom_mlp.train()  # Set model to training mode
    optimizer.zero_grad()  # Zero the gradients
    outputs = custom_mlp(X_train_tensor)  # Forward pass
    loss = criterion(outputs, y_train_tensor)  # Compute the loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights
    
    # Store the loss value
    losses.append(loss.item())
    
    # Print training loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{200}, Loss: {loss.item():.4f}")
        
        # Evaluate the model on test set
        custom_mlp.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            y_pred_logits = custom_mlp(X_test_tensor)  # Get logits
            y_pred = (torch.sigmoid(y_pred_logits) > 0.5).float().numpy()  # Convert to binary predictions
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy at Epoch {epoch + 1}: {accuracy * 100:.2f}%")

# Plot the training loss curve
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

# Final model evaluation
print(f"Final Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=["seen", "unseen"]))
