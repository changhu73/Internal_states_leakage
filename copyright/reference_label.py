import json

input_file_path = '/home/Guangwei/sit/copy-bench/label/literal_labels.json'
output_file_path = '/home/Guangwei/sit/copy-bench/label/literal_reference.json'

with open(input_file_path, 'r', encoding='utf-8') as file:
    json_data = json.load(file)

new_data = []
for entry in json_data:
    new_entry = {
        'reference': entry['reference'],
        'label': 0
    }
    new_data.append(new_entry)

with open(output_file_path, 'w', encoding='utf-8') as file:
    json.dump(new_data, file, ensure_ascii=False, indent=2)

print(f"reference json file saved... {output_file_path}.")
