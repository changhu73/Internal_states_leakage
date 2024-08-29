from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from rouge_score import rouge_scorer
from tqdm import tqdm
import matplotlib.pyplot as plt

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

# Load model and tokenizer
model_name = "EleutherAI/gpt-neox-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
tokenizer.pad_token = tokenizer.eos_token

# Load datasets
seen_df = pd.read_csv("dataset/seen_news.csv")
unseen_df = pd.read_csv("dataset/unseen_news.csv")

seen_texts = seen_df['content'].tolist()
unseen_texts = unseen_df['content'].tolist()

# Function to extract hidden states from texts
def extract_hidden_states(texts, model, tokenizer, batch_size=16):
    hidden_states = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing data batches"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        hidden_states.append(outputs.hidden_states[-1].mean(dim=1).cpu().numpy())
    return np.vstack(hidden_states)

# Extract hidden states for seen and unseen texts
hidden_states_seen = extract_hidden_states(seen_texts, model, tokenizer)
hidden_states_unseen = extract_hidden_states(unseen_texts, model, tokenizer)

X_seen = hidden_states_seen
X_unseen = hidden_states_unseen

y_seen = np.zeros(len(X_seen))
y_unseen = np.ones(len(X_unseen))

# Split data into training and test sets
split_index_seen = int(0.8 * len(X_seen))
split_index_unseen = int(0.8 * len(X_unseen))

X_seen_train = X_seen[:split_index_seen]
X_seen_test = X_seen[split_index_seen:]
y_seen_train = y_seen[:split_index_seen]
y_seen_test = y_seen[split_index_seen:]

X_unseen_train = X_unseen[:split_index_unseen]
X_unseen_test = X_unseen[split_index_unseen:]
y_unseen_train = y_unseen[:split_index_unseen]
y_unseen_test = y_unseen[:split_index_unseen:]

X_train = np.vstack((X_seen_train, X_unseen_train))
X_test = np.vstack((X_seen_test, X_unseen_test))
y_train = np.concatenate((y_seen_train, y_unseen_train))
y_test = np.concatenate((y_seen_test, y_unseen_test))

# Initialize the custom MLP model
input_dim = X_train.shape[1]  # Number of features
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

# Training the MLP model
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

# Plot the training loss curve
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

# First layer evaluation on test set
custom_mlp.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    y_pred_logits = custom_mlp(X_test_tensor)  # Get logits
    y_pred_first_layer = (torch.sigmoid(y_pred_logits) > 0.5).float().numpy()  # Convert to binary predictions

# ROUGE score calculation for the second layer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

def calculate_rouge(text_a, text_b):
    scores = scorer.score(text_a, text_b)
    return scores['rouge1'].fmeasure, scores['rougeL'].fmeasure

# Set thresholds for ROUGE score
rouge1_threshold = 0.5
rougeL_threshold = 0.5

def evaluate_rouge_and_label(text_a, text_b):
    rouge1_fmeasure, rougeL_fmeasure = calculate_rouge(text_a, text_b)
    if rouge1_fmeasure > rouge1_threshold and rougeL_fmeasure > rougeL_threshold:
        return 0  # 侵权
    else:
        return 1  # 未侵权

# Second layer evaluation
final_labels = []
for i, text in enumerate(unseen_texts):  # Here we assume unseen_texts should be compared
    if y_pred_first_layer[len(seen_texts) + i] < 0.5:  # Check first layer prediction
        final_labels.append(0)  # 侵权
    else:
        reference_text = seen_texts[i % len(seen_texts)]  # Example: compare with some reference texts
        rouge_label = evaluate_rouge_and_label(text, reference_text)
        final_labels.append(rouge_label)

# Final evaluation
print(f"Final Model Accuracy: {accuracy_score(y_unseen_test, final_labels) * 100:.2f}%")
print(classification_report(y_unseen_test, final_labels, target_names=["侵权", "未侵权"]))
