# Computer-Vision-







%pip install datasets



from datasets import Dataset
import json
 
def load_translation_dataset(file_path):
    """
    Load and prepare the translation dataset
    """
    # Read the JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Prepare the examples in Llama instruction format
    formatted_data = []
    for example in data['training_data']:
        # Create instruction that includes preservation information
        instruction = f"""Translate from {example['source_lang']} to {example['target_lang']}.
Context: {example['context']}
Preserve these words unchanged and it has to be transliterated to target langauge keeping Abbreviations as is: {', '.join(example['domain_terms'])}
 
Text: {example['source_text']}"""
        formatted_data.append({
            'instruction': instruction,
            'input': '',  # Empty as instruction contains the source text
            'output': example['target_text']
        })
    # Create Hugging Face dataset
    dataset = Dataset.from_list(formatted_data)
    dataset.save_to_disk(r'C:\Users\deekshith.p\EYMP_GitRepo\EPWork\text_translator\Finetuning_Datasets\final_instruction_format_dataset') 
    return dataset



 # Example usage
if __name__ == "__main__":
    # Load the dataset
    dataset = load_translation_dataset(r'C:\Users\deekshith.p\EYMP_GitRepo\EPWork\text_translator\Finetuning_Datasets\translation_dataset_v1.json')
    # Print a sample
    print("\nSample instruction format:")
    print(dataset[0]['instruction'])
    print("\nExpected output:")
    print(dataset[1]['output'])
