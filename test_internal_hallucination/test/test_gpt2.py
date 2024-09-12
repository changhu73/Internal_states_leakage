from transformers import AutoModel, AutoTokenizer
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

tokenizer.pad_token = tokenizer.eos_token

# Load data
seen_df = pd.read_csv("dataset/seen_news.csv")
unseen_df = pd.read_csv("dataset/unseen_news.csv")

seen_texts = seen_df['content'].tolist()
unseen_texts = unseen_df['content'].tolist()

# Function to extract hidden states
def extract_hidden_states(texts, model, tokenizer):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use mean of all hidden states instead of the last one
    hidden_states = outputs.hidden_states[-1].mean(dim=1).numpy()
    return hidden_states

# Extract hidden states
hidden_states_seen = extract_hidden_states(seen_texts, model, tokenizer)
hidden_states_unseen = extract_hidden_states(unseen_texts, model, tokenizer)

# Prepare data for training
X_seen = hidden_states_seen
X_unseen = hidden_states_unseen

y_seen = np.zeros(len(X_seen))
y_unseen = np.ones(len(X_unseen))

X = np.vstack((X_seen, X_unseen))
y = np.concatenate((y_seen, y_unseen))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the MLP model with tuned parameters
mlp = MLPClassifier(hidden_layer_sizes=(512, 256), activation='relu', max_iter=200, random_state=42, learning_rate_init=0.001)

# Cross-validation to assess the model performance
cv_scores = cross_val_score(mlp, X_train, y_train, cv=5)  # 5-fold cross-validation
print(f"交叉验证准确率: {cv_scores.mean() * 100:.2f}%")

# Fit the model on the training data
mlp.fit(X_train, y_train)

# Predict on the test set
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy * 100:.2f}%")

# Classification report
print(classification_report(y_test, y_pred, target_names=["seen", "unseen"]))

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=y_test, cmap='viridis')
plt.title('PCA降维后的特征分布')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.colorbar()
plt.show()