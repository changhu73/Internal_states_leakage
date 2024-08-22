from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 加载 BERT 模型和分词器
model_name = "bert-base-uncased"  # 或者选择其他 BERT 模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 读取数据集
seen_df = pd.read_csv("dataset/seen_news.csv")
unseen_df = pd.read_csv("dataset/unseen_news.csv")

seen_texts = seen_df['content'].tolist()
unseen_texts = unseen_df['content'].tolist()

# 提取标签
y_seen = [0] * len(seen_texts)  # 0 for seen
y_unseen = [1] * len(unseen_texts)  # 1 for unseen

# 合并数据
texts = seen_texts + unseen_texts
labels = y_seen + y_unseen

# 数据集切分
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 数据预处理
train_encodings = tokenizer(X_train, truncation=True, padding=True, return_tensors="pt", max_length=512)
test_encodings = tokenizer(X_test, truncation=True, padding=True, return_tensors="pt", max_length=512)

# 创建 Tensor 数据集
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

# 模型训练
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    num_train_epochs=3,              # 训练周期
    per_device_train_batch_size=8,   # 每个设备的训练批量
    per_device_eval_batch_size=16,    # 每个设备的评估批量
    warmup_steps=500,                 # 预热步数
    weight_decay=0.01,               # 权重衰减
    logging_dir='./logs',            # 日志文件夹
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
