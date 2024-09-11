import json
import numpy as np

with open('/home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scores/scores-literal-copying.literal.prompt1.Meta-Llama-3.1-8B.greedy.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

scores = [entry['score_rouge_1'] for entry in data]

# threshold = np.median(scores)
# print(threshold)

group1 = [entry for entry in data if entry['score_rouge_1'] > 0.27]
group2 = [entry for entry in data if entry['score_rouge_1'] < 0.09]

# infringement
for entry in group1:
    entry['label'] = 0

# non-infringement
for entry in group2:
    entry['label'] = 1

print(f'Group infringement: {len(group1)} entries')
print(f'Group non-infringement: {len(group2)} entries')

with open('/home/guangwei/LLM-COPYRIGHT/copyright_newVersion/test_division/literal.infringement.json', 'w', encoding='utf-8') as file:
    json.dump(group1, file, ensure_ascii=False, indent=4)

with open('/home/guangwei/LLM-COPYRIGHT/copyright_newVersion/test_division/literal.non_infringement.json', 'w', encoding='utf-8') as file:
    json.dump(group2, file, ensure_ascii=False, indent=4)
