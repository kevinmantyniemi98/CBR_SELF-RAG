from utility_functions import load_jsonl, save_to_jsonl, preprocess_text, evaluate_utility, relevance_decision, calculate_text_similarity

# Example usage scenario
data = load_jsonl('data/raw_selfrag_data.jsonl')

augmented_data = []
for item in data:
    # Preprocess item text
    processed_text = preprocess_text(item['instruction'])
    
    # Evaluate utility for simple demonstration
    utility_label = evaluate_utility(item.get('utility', 0))

    # Add augmented data
    augmented_data.append({
        'instruction': item['instruction'],
        'output': item['output'],
        'processed_text': processed_text,
        'utility_label': utility_label
        # Add further processing results as needed
    })

save_to_jsonl(augmented_data, 'data/augmented_cbr_data.jsonl')