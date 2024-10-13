from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch

model_name = "meta-llama/Meta-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# # 输入文本
# input_text = "What is the capital of France? "

# # 使用分词器将文本编码为模型可以理解的格式
# inputs = tokenizer.encode(input_text, return_tensors="pt")

# # 生成文本
# # max_length 控制生成文本的最大长度
# # num_return_sequences 控制生成的序列数量
# # no_repeat_ngram_size, do_sample, top_k, top_p 等参数可以根据需要进行调整
# output_sequences = model.generate(
#     inputs,
#     max_length=50,
#     num_return_sequences=1,
#     no_repeat_ngram_size=0,
#     do_sample=True,
#     top_k=50,
#     top_p=0.95
# )

# # 将生成的序列解码回文本
# generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

# print(generated_text)

texts = ["Hello, how are you?", "This is a test sentence."]
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits
predicted_tokens = torch.argmax(logits, dim=-1) 
decoded_texts = [tokenizer.decode(pred) for pred in predicted_tokens]

print("Original Texts: ", texts)
print("Decoded Texts from Logits: ", decoded_texts)
