import json

def clean_output(entry):
    input_text = entry['input']
    output_text = entry['output']
    
    pos = output_text.find(input_text)
    if pos != -1:
        cleaned_output = output_text[pos + len(input_text):].strip()
    else:
        cleaned_output = output_text.strip()
    
    return cleaned_output

with open('/home/Guangwei/sit/copy-bench/generate/qa_outputs.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

for entry in data:
    entry['output'] = clean_output(entry)

with open('/home/Guangwei/sit/copy-bench/cleaned/qa_data.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=2, ensure_ascii=False)

print("Data cleaning completed successfully!")
