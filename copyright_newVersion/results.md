9. internal state selection

(1) 对每个文本批次进行处理并取最后一层的平均值，最普通版本
```python
def extract_hidden_states(texts, model, tokenizer, batch_size=4):
    hidden_states = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing data batches"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        hidden_states.append(outputs.hidden_states[-1].mean(dim=1).cpu().numpy())
    return np.vstack(hidden_states)
```


(2) 选择不同层：指定layer_index
``` python
def extract_hidden_states(texts, model, tokenizer, layer_index=-1, batch_size=4):
    hidden_states = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing data batches"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        # 修改: 提取指定的层
        selected_layer_hidden_state = outputs.hidden_states[layer_index].mean(dim=1).cpu().numpy()
        hidden_states.append(selected_layer_hidden_state)
    return np.vstack(hidden_states)
```

(3) 提取多个层的隐藏状态并进行融合, 可以去函数里直接修改layers的元素值
```python
def extract_hidden_states(texts, model, tokenizer, layers=None, batch_size=4):
    hidden_states = []
    if layers is None:
        layers = [-4, -3, -2, -1]  # 默认提取最后4层
        
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing data batches"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)

        # 提取并融合指定层的隐藏状态
        selected_layers_hidden_states = [outputs.hidden_states[layer].mean(dim=1) for layer in layers]
        fused_hidden_state = torch.cat(selected_layers_hidden_states, dim=-1).cpu().numpy()  # 拼接多个层
        hidden_states.append(fused_hidden_state)
        
    return np.vstack(hidden_states)
```

(4) 隐藏状态的池化策略改进，修改pooling_strategy
```python
def extract_hidden_states(texts, model, tokenizer, pooling_strategy='mean', batch_size=4):
    hidden_states = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing data batches"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        layer_hidden_states = outputs.hidden_states[-1]

        if pooling_strategy == 'mean':
            pooled_hidden_state = layer_hidden_states.mean(dim=1).cpu().numpy()
        elif pooling_strategy == 'max':
            pooled_hidden_state = layer_hidden_states.max(dim=1).values.cpu().numpy()
        elif pooling_strategy == 'cls':
            pooled_hidden_state = layer_hidden_states[:, 0, :].cpu().numpy()  # First token (CLS)
        else:
            raise ValueError("Unknown pooling strategy")

        hidden_states.append(pooled_hidden_state)
    return np.vstack(hidden_states)

```


(5) 降维技术（如PCA、t-SNE）来减少特征的维度，或者通过归一化操作使特征更加标准化(apply_pca=True进行降维)
```python
from sklearn.decomposition import PCA

def extract_hidden_states(texts, model, tokenizer, apply_pca=False, n_components=50, batch_size=4):
    hidden_states = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing data batches"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        hidden_state = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
        hidden_states.append(hidden_state)

    hidden_states = np.vstack(hidden_states)
    
    # 额外处理步骤: PCA降维
    if apply_pca:
        pca = PCA(n_components=n_components)
        hidden_states = pca.fit_transform(hidden_states)
        print(f"Hidden states reduced to {n_components} dimensions using PCA.")
        
    return hidden_states
```