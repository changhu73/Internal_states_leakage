import json
import numpy as np

with open('/home/Guangwei/sit/copy-bench/scores/scores-literal-copying.literal.prompt1.Llama-2-7b-hf.greedy.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

scores = [entry['score_rouge_1'] for entry in data]

threshold = np.median(scores)
print(threshold)

group1 = [entry for entry in data if entry['score_rouge_1'] > threshold]
group2 = [entry for entry in data if entry['score_rouge_1'] <= threshold]

for entry in group1:
    entry['label'] = 0

for entry in group2:
    entry['label'] = 1

print(f'Group 1: {len(group1)} entries')
print(f'Group 2: {len(group2)} entries')

with open('/home/Guangwei/sit/copy-bench/test_division/literal.infringement.json', 'w', encoding='utf-8') as file:
    json.dump(group1, file, ensure_ascii=False, indent=4)

with open('/home/Guangwei/sit/copy-bench/test_division/literal.non_infringement.json', 'w', encoding='utf-8') as file:
    json.dump(group2, file, ensure_ascii=False, indent=4)
