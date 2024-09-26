# import argparse
# import torch
# import torch.nn as nn
# import numpy as np
# from sklearn.metrics import accuracy_score, classification_report
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import json

# class CustomMLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(CustomMLP, self).__init__()
#         self.down = nn.Linear(input_dim, hidden_dim)
#         self.gate = nn.Linear(input_dim, hidden_dim)
#         self.up = nn.Linear(hidden_dim, 1)

#         self.activation = nn.SiLU()

#     def forward(self, x):
#         down_output = self.down(x)
#         gate_output = self.gate(x)
#         gated_output = down_output * self.activation(gate_output)
#         return self.up(gated_output)

# # 1.取最后一层，普通版本
# def extract_hidden_states(texts, model, tokenizer, batch_size=4):
#     hidden_states = []
#     for i in tqdm(range(0, len(texts), batch_size), desc="Processing data batches"):
#         batch_texts = texts[i:i + batch_size]
#         inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         hidden_states.append(outputs.hidden_states[-1].mean(dim=1).cpu().numpy())
#     return np.vstack(hidden_states)

# # # 2.选择不同层：指定layer_index
# # def extract_hidden_states(texts, model, tokenizer, layer_index=-1, batch_size=4):
# #     hidden_states = []
# #     for i in tqdm(range(0, len(texts), batch_size), desc="Processing data batches"):
# #         batch_texts = texts[i:i + batch_size]
# #         inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
# #         with torch.no_grad():
# #             outputs = model(**inputs)
# #         selected_layer_hidden_state = outputs.hidden_states[layer_index].mean(dim=1).cpu().numpy()
# #         hidden_states.append(selected_layer_hidden_state)
# #     return np.vstack(hidden_states)

# # # 3. 提取多个层的隐藏状态并进行融合, 可以去函数里直接修改layers的元素值，真实测出来会高一些
# # def extract_hidden_states(texts, model, tokenizer, layers=None, batch_size=4):
# #     hidden_states = []
# #     if layers is None:
# #         layers = [-4, -3, -2, -1]  # 默认提取最后4层
        
# #     for i in tqdm(range(0, len(texts), batch_size), desc="Processing data batches"):
# #         batch_texts = texts[i:i + batch_size]
# #         inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
# #         with torch.no_grad():
# #             outputs = model(**inputs)

# #         # 提取并融合指定层的隐藏状态
# #         selected_layers_hidden_states = [outputs.hidden_states[layer].mean(dim=1) for layer in layers]
# #         fused_hidden_state = torch.cat(selected_layers_hidden_states, dim=-1).cpu().numpy()  # 拼接多个层
# #         hidden_states.append(fused_hidden_state)
        
# #     return np.vstack(hidden_states)

# # # 4.隐藏状态的池化策略改进，修改pooling_strategy
# # def extract_hidden_states(texts, model, tokenizer, pooling_strategy='cls', batch_size=4):
# #     hidden_states = []
# #     for i in tqdm(range(0, len(texts), batch_size), desc="Processing data batches"):
# #         batch_texts = texts[i:i + batch_size]
# #         inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
# #         with torch.no_grad():
# #             outputs = model(**inputs)
# #         layer_hidden_states = outputs.hidden_states[-1]

# #         if pooling_strategy == 'mean':
# #             pooled_hidden_state = layer_hidden_states.mean(dim=1).cpu().numpy()
# #         elif pooling_strategy == 'max':
# #             pooled_hidden_state = layer_hidden_states.max(dim=1).values.cpu().numpy()
# #         elif pooling_strategy == 'cls':
# #             pooled_hidden_state = layer_hidden_states[:, 0, :].cpu().numpy()  # First token (CLS)
# #         else:
# #             raise ValueError("Unknown pooling strategy")

# #         hidden_states.append(pooled_hidden_state)
# #     return np.vstack(hidden_states)

# # # 5.降维技术（如PCA、t-SNE）来减少特征的维度，或者通过归一化操作使特征更加标准化(apply_pca=True进行降维) 
# # from sklearn.decomposition import PCA

# # def extract_hidden_states(texts, model, tokenizer, apply_pca=True, n_components=50, batch_size=4):
# #     hidden_states = []
# #     for i in tqdm(range(0, len(texts), batch_size), desc="Processing data batches"):
# #         batch_texts = texts[i:i + batch_size]
# #         inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
# #         with torch.no_grad():
# #             outputs = model(**inputs)
# #         hidden_state = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
# #         hidden_states.append(hidden_state)

# #     hidden_states = np.vstack(hidden_states)
    
# #     if apply_pca:
# #         pca = PCA(n_components=n_components)
# #         hidden_states = pca.fit_transform(hidden_states)
# #         print(f"Hidden states reduced to {n_components} dimensions using PCA.")
        
# #     return hidden_states

# # 6.TSNE降维
# from sklearn.manifold import TSNE
# def extract_hidden_states(texts, model, tokenizer, apply_tsne=True, n_components=2, batch_size=4):
#     hidden_states = []
#     for i in tqdm(range(0, len(texts), batch_size), desc="Processing data batches"):
#         batch_texts = texts[i:i + batch_size]
#         inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         hidden_state = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
#         hidden_states.append(hidden_state)

#     hidden_states = np.vstack(hidden_states)
    
#     if apply_tsne:
#         tsne = TSNE(n_components=n_components, random_state=42)
#         hidden_states = tsne.fit_transform(hidden_states)
#         print(f"Hidden states reduced to {n_components} dimensions using t-SNE.")
        
#     return hidden_states

# # 7.
# import umap

# def extract_hidden_states(texts, model, tokenizer, apply_umap=True, n_components=2, batch_size=4):
#     hidden_states = []
#     for i in tqdm(range(0, len(texts), batch_size), desc="Processing data batches"):
#         batch_texts = texts[i:i + batch_size]
#         inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         hidden_state = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
#         hidden_states.append(hidden_state)

#     hidden_states = np.vstack(hidden_states)
    
#     if apply_umap:
#         reducer = umap.UMAP(n_components=n_components, random_state=42)
#         hidden_states = reducer.fit_transform(hidden_states)
#         print(f"Hidden states reduced to {n_components} dimensions using UMAP.")
        
#     return hidden_states


# def load_data(non_infringement_file, infringement_file):
#     with open(non_infringement_file, 'r', encoding='utf-8') as file:
#         non_infringement_json_data = json.load(file)

#     non_infringement_outputs = [entry['input'] for entry in non_infringement_json_data]
#     y_non_infringement = [1] * len(non_infringement_outputs)

#     with open(infringement_file, 'r', encoding='utf-8') as file:
#         infringement_json_data = json.load(file)

#     infringement_outputs = [entry['input'] for entry in infringement_json_data]
#     y_infringement = [0] * len(infringement_outputs)

#     return non_infringement_outputs, y_non_infringement, infringement_outputs, y_infringement


# def train_model(X_train, y_train, X_test, y_test, input_dim, hidden_dim, epochs=500, lr=0.001, checkpoint_path="/home/guangwei/LLM-COPYRIGHT/copyright_newVersion/models/best_model.pth"):
#     custom_mlp = CustomMLP(input_dim, hidden_dim)
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = torch.optim.Adam(custom_mlp.parameters(), lr=lr)

#     X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#     y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

#     best_accuracy = -float('inf')  # Initialize the best accuracy to negative infinity
#     best_model_state = None  # Store the state of the best model
#     best_epoch = 0  # Track the epoch with the best accuracy
#     losses = []

#     for epoch in tqdm(range(epochs), desc="Training Epochs"):
#         custom_mlp.train()
#         optimizer.zero_grad()
#         outputs = custom_mlp(X_train_tensor)
#         loss = criterion(outputs, y_train_tensor)
#         loss.backward()
#         optimizer.step()
#         losses.append(loss.item())
        
#         if (epoch + 1) % 10 == 0:
#             print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
            
#             custom_mlp.eval()
#             X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
#             with torch.no_grad():
#                 y_pred_logits = custom_mlp(X_test_tensor)
#                 y_pred = (torch.sigmoid(y_pred_logits) > 0.5).float().numpy()
            
#             accuracy = accuracy_score(y_test, y_pred)
#             print(f"Test Accuracy at Epoch {epoch + 1}: {accuracy * 100:.2f}%")
            
#             # Compute precision, recall, F1-score, etc.
#             report = classification_report(y_test, y_pred, target_names=["infringement", "non_infringement"])
#             print(f"Classification Report at Epoch {epoch + 1}:\n{report}")

#             # Save model if this epoch's accuracy is the best
#             if accuracy > best_accuracy:
#                 best_accuracy = accuracy
#                 best_model_state = custom_mlp.state_dict()  # Save the best model state
#                 best_epoch = epoch + 1
#                 torch.save(best_model_state, checkpoint_path)
#                 print(f"New best model saved with accuracy {best_accuracy * 100:.2f}% at epoch {best_epoch}")
#                 print(f"Best Classification Report at Epoch {best_epoch}:\n{report}")

#     # After all epochs, load the best model for final evaluation
#     custom_mlp.load_state_dict(torch.load(checkpoint_path))

#     plt.figure(figsize=(10, 5))
#     plt.plot(losses, label='Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training Loss Curve')
#     plt.legend()
#     plt.show()

#     print(f"Best Model was saved at epoch {best_epoch} with accuracy {best_accuracy * 100:.2f}%")
#     return custom_mlp, losses, best_accuracy

# def save_checkpoint(model, optimizer, epoch, loss, filepath):
#     checkpoint = {
#         'epoch': epoch + 1,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss
#     }
#     torch.save(checkpoint, filepath)
#     print(f"Checkpoint saved to '{filepath}'.")

# def main(args):
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=512)
#     model = AutoModelForCausalLM.from_pretrained(args.model_name, output_hidden_states=True)
#     tokenizer.pad_token = tokenizer.eos_token

#     non_infringement_outputs, y_non_infringement, infringement_outputs, y_infringement = load_data(
#         args.non_infringement_file, args.infringement_file
#     )
    
#     y_non_infringement = np.array(y_non_infringement)
#     y_infringement = np.array(y_infringement)

#     print("Extracting hidden states for non_infringement texts...")
#     X_non_infringement = extract_hidden_states(non_infringement_outputs, model, tokenizer)

#     print("Extracting hidden states for infringement texts...")
#     X_infringement = extract_hidden_states(infringement_outputs, model, tokenizer)

#     split_index_non_infringement = int(0.8 * len(X_non_infringement))
#     X_non_infringement_train = X_non_infringement[:split_index_non_infringement]
#     X_non_infringement_test = X_non_infringement[split_index_non_infringement:]
#     y_non_infringement_train = y_non_infringement[:split_index_non_infringement]
#     y_non_infringement_test = y_non_infringement[split_index_non_infringement:]

#     split_index_infringement = int(0.8 * len(X_infringement))
#     X_infringement_train = X_infringement[:split_index_infringement]
#     X_infringement_test = X_infringement[split_index_infringement:]
#     y_infringement_train = y_infringement[:split_index_infringement]
#     y_infringement_test = y_infringement[split_index_infringement:]

#     X_train = np.vstack((X_non_infringement_train, X_infringement_train))
#     X_test = np.vstack((X_non_infringement_test, X_infringement_test))
#     y_train = np.concatenate((y_non_infringement_train, y_infringement_train))
#     y_test = np.concatenate((y_non_infringement_test, y_infringement_test))

#     print("Data successfully split into training and test sets.")

#     input_dim = X_train.shape[1]
#     hidden_dim = 256 

#     custom_mlp, losses, accuracy = train_model(X_train, y_train, X_test, y_test, input_dim, hidden_dim)

#     save_checkpoint(custom_mlp, torch.optim.Adam(custom_mlp.parameters()), len(losses), losses[-1], args.checkpoint_file)

#     print(f"Final Model Accuracy: {accuracy * 100:.2f}%")
#     print(classification_report(y_test, (torch.sigmoid(torch.tensor(custom_mlp(torch.tensor(X_test, dtype=torch.float32)))) > 0.5).float().numpy(), target_names=["infringement", "non_infringement"]))

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train a Custom MLP for infringement detection.")
#     parser.add_argument('--model_name', type=str, help='Name of the pretrained model.', default='meta-llama/Llama-2-7b-hf')
#     parser.add_argument('--non_infringement_file', type=str, help='Path to the non-infringement data file.', default='/home/guangwei/LLM-COPYRIGHT/copyright_newVersion/test_division/extra.non_infringement.json')
#     parser.add_argument('--infringement_file', type=str, help='Path to the infringement data file.', default='/home/guangwei/LLM-COPYRIGHT/copyright_newVersion/test_division/extra.infringement.json')
#     parser.add_argument('--checkpoint_file', type=str, help='Path to save the model checkpoint.', default='/home/guangwei/LLM-COPYRIGHT/copyright_newVersion/models/custom_mlp_model.pth')

#     args = parser.parse_args()
#     main(args)

# # add reference embedding
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
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

# def extract_hidden_states(texts, model, tokenizer, batch_size=4):
#     hidden_states = []
#     for i in tqdm(range(0, len(texts), batch_size), desc="Processing data batches"):
#         batch_texts = texts[i:i + batch_size]
#         inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         hidden_states.append(outputs.hidden_states[-1].mean(dim=1).cpu().numpy())
#     return np.vstack(hidden_states)

# def extract_reference_embeddings(references, model, tokenizer, batch_size=4):
#     embeddings = []
#     for i in tqdm(range(0, len(references), batch_size), desc="Processing references"):
#         batch_references = references[i:i + batch_size]
#         inputs = tokenizer(batch_references, return_tensors="pt", padding=True, truncation=True)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         embeddings.append(outputs.pooler_output.cpu().numpy())
#     return np.vstack(embeddings)

def extract_hidden_states(texts, model, tokenizer, batch_size=4):
    hidden_states = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing data batches"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=1)  # 限制为前5个token
        input_ids = inputs['input_ids'][:, :1]  # 取前5个token的ID
        attention_mask = inputs['attention_mask'][:, :1]  # 取前5个token的attention mask

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # 只考虑前5个token的 hidden states，取最后一层
        last_hidden_state = outputs.hidden_states[-1][:, :1, :]  # (batch_size, 5, hidden_dim)

        # 对前5个token的 hidden states 取均值
        hidden_state_mean = last_hidden_state.mean(dim=1).cpu().numpy()  # (batch_size, hidden_dim)
        
        hidden_states.append(hidden_state_mean)

    return np.vstack(hidden_states)

def extract_reference_embeddings(references, model, tokenizer, batch_size=4):
    embeddings = []
    for i in tqdm(range(0, len(references), batch_size), desc="Processing references"):
        batch_references = references[i:i + batch_size]
        inputs = tokenizer(batch_references, return_tensors="pt", padding=True, truncation=True, max_length=1)  # 限制为前5个token
        input_ids = inputs['input_ids'][:, :1]  # 取前5个token的ID
        attention_mask = inputs['attention_mask'][:, :1]  # 取前5个token的attention mask

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # 对 pooler_output 取前5个token，BERT 的 pooler_output 是整句话的嵌入，但我们只需要对齐前5个token
        pooler_output = outputs.pooler_output.cpu().numpy()  # (batch_size, hidden_dim)
        
        embeddings.append(pooler_output)

    return np.vstack(embeddings)


def load_data(non_infringement_file, infringement_file):
    with open(non_infringement_file, 'r', encoding='utf-8') as file:
        non_infringement_json_data = json.load(file)

    non_infringement_outputs = [entry['input'] for entry in non_infringement_json_data]
    non_infringement_references = [entry['reference'] for entry in non_infringement_json_data]
    y_non_infringement = [1] * len(non_infringement_outputs)

    with open(infringement_file, 'r', encoding='utf-8') as file:
        infringement_json_data = json.load(file)

    infringement_outputs = [entry['input'] for entry in infringement_json_data]
    infringement_references = [entry['reference'] for entry in infringement_json_data]
    y_infringement = [0] * len(infringement_outputs)

    return non_infringement_outputs, non_infringement_references, y_non_infringement, infringement_outputs, infringement_references, y_infringement

def train_model(X_train, y_train, X_test, y_test, input_dim, hidden_dim, epochs=500, lr=0.001, checkpoint_path="/home/guangwei/LLM-COPYRIGHT/copyright_newVersion/models/best_model.pth"):
    custom_mlp = CustomMLP(input_dim, hidden_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(custom_mlp.parameters(), lr=lr)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    best_accuracy = -float('inf')
    best_model_state = None
    best_epoch = 0
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
            
            report = classification_report(y_test, y_pred, target_names=["infringement", "non_infringement"])
            print(f"Classification Report at Epoch {epoch + 1}:\n{report}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = custom_mlp.state_dict()
                best_epoch = epoch + 1
                torch.save(best_model_state, checkpoint_path)
                print(f"New best model saved with accuracy {best_accuracy * 100:.2f}% at epoch {best_epoch}")
                print(f"Best Classification Report at Epoch {best_epoch}:\n{report}")

    custom_mlp.load_state_dict(torch.load(checkpoint_path))

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

    print(f"Best Model was saved at epoch {best_epoch} with accuracy {best_accuracy * 100:.2f}%")
    return custom_mlp, losses, best_accuracy

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=512)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, output_hidden_states=True)
    tokenizer.pad_token = tokenizer.eos_token
    bert_tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
    bert_model = AutoModel.from_pretrained('google-bert/bert-base-uncased')
    bert_tokenizer.pad_token = tokenizer.eos_token

    non_infringement_outputs, non_infringement_references, y_non_infringement, infringement_outputs, infringement_references, y_infringement = load_data(
        args.non_infringement_file, args.infringement_file
    )

    y_non_infringement = np.array(y_non_infringement)
    y_infringement = np.array(y_infringement)

    print("Extracting hidden states for non_infringement texts...")
    X_non_infringement = extract_hidden_states(non_infringement_outputs, model, tokenizer)
    print("Extracting reference embeddings for non_infringement texts...")
    reference_embeddings_non_infringement = extract_reference_embeddings(non_infringement_references, bert_model, bert_tokenizer)
    X_non_infringement_combined = np.hstack([X_non_infringement, reference_embeddings_non_infringement])

    print("Extracting hidden states for infringement texts...")
    X_infringement = extract_hidden_states(infringement_outputs, model, tokenizer)
    print("Extracting reference embeddings for infringement texts...")
    reference_embeddings_infringement = extract_reference_embeddings(infringement_references, bert_model, bert_tokenizer)
    X_infringement_combined = np.hstack([X_infringement, reference_embeddings_infringement])

    split_index_non_infringement = int(0.8 * len(X_non_infringement_combined))
    X_non_infringement_train = X_non_infringement_combined[:split_index_non_infringement]
    X_non_infringement_test = X_non_infringement_combined[split_index_non_infringement:]
    y_non_infringement_train = y_non_infringement[:split_index_non_infringement]
    y_non_infringement_test = y_non_infringement[split_index_non_infringement:]

    split_index_infringement = int(0.8 * len(X_infringement_combined))
    X_infringement_train = X_infringement_combined[:split_index_infringement]
    X_infringement_test = X_infringement_combined[split_index_infringement:]
    y_infringement_train = y_infringement[:split_index_infringement]
    y_infringement_test = y_infringement[split_index_infringement:]

    X_train = np.vstack((X_non_infringement_train, X_infringement_train))
    X_test = np.vstack((X_non_infringement_test, X_infringement_test))
    y_train = np.concatenate((y_non_infringement_train, y_infringement_train))
    y_test = np.concatenate((y_non_infringement_test, y_infringement_test))

    input_dim = X_train.shape[1]
    hidden_dim = 256 
    print(f"Training MLP model with input_dim={input_dim} and hidden_dim={hidden_dim}")

    best_model, losses, best_accuracy = train_model(X_train, y_train, X_test, y_test, input_dim, hidden_dim)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Custom MLP for infringement detection.")
    parser.add_argument('--model_name', type=str, help='Name of the pretrained model.', default='meta-llama/Meta-Llama-3.1-8B')
    parser.add_argument('--non_infringement_file', type=str, help='Path to the non-infringement data file.', default='/home/guangwei/LLM-COPYRIGHT/copyright_newVersion/test_division/extra.non_infringement.json')
    parser.add_argument('--infringement_file', type=str, help='Path to the infringement data file.', default='/home/guangwei/LLM-COPYRIGHT/copyright_newVersion/test_division/extra.infringement.json')
    parser.add_argument('--checkpoint_file', type=str, help='Path to save the model checkpoint.', default='/home/guangwei/LLM-COPYRIGHT/copyright_newVersion/models/custom_mlp_model.pth')

    args = parser.parse_args()
    main(args)
