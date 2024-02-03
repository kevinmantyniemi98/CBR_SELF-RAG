import jsonlines
import pandas as pd

def load_data(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data

def augment_data(entry):
    """
    Process a single entry and decide on Retain, Revise, or Recall.
    The logic here is greatly simplified and should be replaced with your specific criteria.
    """
    augmented_entry = entry.copy()
    # Example of simplified decision logic based on utility score
    utility_score = augmented_entry.get('utility', 0)
    if utility_score == 5:
        # High utility -> Retain
        augmented_entry['retain'] = True
        augmented_entry['revise'] = False
    elif utility_score < 3:
        # Low utility -> Revise
        augmented_entry['retain'] = False
        augmented_entry['revise'] = True
    else:
        # Medium utility -> potential Recall check (not implemented here)
        augmented_entry['retain'] = False
        augmented_entry['revise'] = False  # Assume further analysis needed for Recall logic

    # Add more sophisticated logic here based on relevance (ISREL), support (ISSUP), etc.
    return augmented_entry

def process_dataset(data):
    """
    Process each entry in the dataset.
    """
    augmented_data = [augment_data(entry) for entry in data]
    return augmented_data

def save_augmented_data(augmented_data, output_file):
    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(augmented_data)

def main():
    input_file_path = 'data/raw_selfrag_data.jsonl'
    output_file_path = 'data/augmented_cbr_data.jsonl'

    # Load the dataset
    data = load_data(input_file_path)
    print(f"Loaded {len(data)} entries from the dataset.")

    # Process and augment the dataset with "Retain", "Revise", "Recall" logic
    augmented_data = process_dataset(data)
    print(f"Processed {len(augmented_data)} entries.")

    # Save the augmented dataset
    save_augmented_data(augmented_data, output_file_path)
    print(f"Augmented dataset saved to {output_file_path}")

if __name__ == "__main__":
    main()
    
    