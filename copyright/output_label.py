import json

input_file_path = '/home/Guangwei/sit/copy-bench/label/literal_labels.json'
output_file_path = '/home/Guangwei/sit/copy-bench/label/literal_output.json'

with open(input_file_path, 'r', encoding='utf-8') as file:
    json_data = json.load(file)

new_data = []
for idx, entry in enumerate(json_data, start=1):
    new_entry = {
        'id': idx,
        'output': entry['output'],
        'label': entry['label']
    }
    new_data.append(new_entry)

with open(output_file_path, 'w', encoding='utf-8') as file:
    json.dump(new_data, file, ensure_ascii=False, indent=2)

print(f"Reference JSON file saved... {output_file_path}.")
