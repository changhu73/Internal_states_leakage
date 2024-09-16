import json
import numpy as np
import argparse

def process_data(scores_file, infringement_output, non_infringement_output):
    with open(scores_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # scores = [entry['score_rouge_1'] for entry in data]
    # threshold = np.median(scores)
    # print(f'Median threshold: {threshold}')

    group1 = [entry for entry in data if entry['score_rouge_1'] > 0.27]
    group2 = [entry for entry in data if entry['score_rouge_1'] < 0.1]

    for entry in group1:
        entry['label'] = 0

    for entry in group2:
        entry['label'] = 1

    print(f'Group infringement: {len(group1)} entries')
    print(f'Group non-infringement: {len(group2)} entries')

    with open(infringement_output, 'w', encoding='utf-8') as file:
        json.dump(group1, file, ensure_ascii=False, indent=4)

    with open(non_infringement_output, 'w', encoding='utf-8') as file:
        json.dump(group2, file, ensure_ascii=False, indent=4)

    print(f'Data saved to {infringement_output} and {non_infringement_output}')

def main(args):
    process_data(args.scores_file, args.infringement_output, args.non_infringement_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process scores and divide into infringement and non-infringement groups.")
    parser.add_argument('--scores_file', type=str, help='Path to the input scores JSON file.', default='/home/guangwei/LLM-COPYRIGHT/copyright_newVersion/scores/scores-literal-copying.literal.prompt1.Meta-Llama-3.1-8B.greedy.json')
    parser.add_argument('--infringement_output', type=str, help='Path to save the infringement group JSON.', default='/home/guangwei/LLM-COPYRIGHT/copyright_newVersion/test_division/literal.infringement.json')
    parser.add_argument('--non_infringement_output', type=str, help='Path to save the non-infringement group JSON.', default='/home/guangwei/LLM-COPYRIGHT/copyright_newVersion/test_division/literal.non_infringement.json')

    args = parser.parse_args()
    main(args)
