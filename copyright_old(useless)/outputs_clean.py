import json
import re

def clean_output(entry):
    input_text = entry['input']
    output_text = entry['output']
    
    pos = output_text.find(input_text)
    if pos != -1:
        cleaned_output = output_text[pos + len(input_text):].strip()
    else:
        cleaned_output = output_text.strip()
    
    return cleaned_output

def truncate_to_reference_length(entry):
    reference = entry['reference']
    output_text = entry['output']

    ref_length = len(reference)
    
    sentence_endings = [m.end() for m in re.finditer(r'[.!?]', output_text)]
    
    if sentence_endings:
        closest_ending = min(sentence_endings, key=lambda x: abs(x - ref_length))
        truncated_output = output_text[:closest_ending].strip()
    else:
        truncated_output = output_text.strip()
    
    return truncated_output

with open('/home/Guangwei/sit/copy-bench/generate/literal_outputs.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

for entry in data:
    entry['output'] = clean_output(entry)
    entry['output'] = truncate_to_reference_length(entry)

with open('/home/Guangwei/sit/copy-bench/cleaned/literal_data.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=2, ensure_ascii=False)

print("Data cleaning and truncation completed successfully!")
