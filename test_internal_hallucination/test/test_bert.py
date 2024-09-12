from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

seen_df = pd.read_csv("dataset/seen_news.csv")
unseen_df = pd.read_csv("dataset/unseen_news.csv")

seen_texts = seen_df['content'].tolist()
unseen_texts = unseen_df['content'].tolist()

y_seen = [0] * len(seen_texts)
y_unseen = [1] * len(unseen_texts)

texts = seen_texts + unseen_texts
labels = y_seen + y_unseen

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

train_encodings = tokenizer(X_train, truncation=True, padding=True, return_tensors="pt", max_length=512)
test_encodings = tokenizer(X_test, truncation=True, padding=True, return_tensors="pt", max_length=512)

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, y_train)
test_dataset = NewsDataset(test_encodings, y_test)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./test',        
    num_train_epochs=3,            
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=16,   
    warmup_steps=500,             
    weight_decay=0.01,            
    logging_dir='./logs',      
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

predictions = trainer.predict(test_dataset)

logits = predictions.predictions
probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1)

for i, prob in enumerate(probabilities):
    print(f"样本 {i + 1} 的预测概率: 类别 0: {prob[0].item():.4f}, 类别 1: {prob[1].item():.4f}")

y_pred = np.argmax(predictions.predictions, axis=1)

accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))
