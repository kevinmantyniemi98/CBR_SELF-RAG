import json

def load_dataset(file_path):
    """
    Load a dataset from a given file path. Supports JSON and JSONLines formats.
    
    Args:
    file_path (str): The path to the dataset file.
    
    Returns:
    list: A list of data items loaded from the file.
    """
    data = []
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        elif file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    data.append(json.loads(line.strip()))
        else:
            print(f"Unsupported file format for {file_path}")
    except Exception as e:
        print(f"Failed to load dataset from {file_path}: {e}")
    return data

def preprocess_data(data):
    """
    Preprocess the dataset for consistency. This function cleans text data by
    removing unnecessary spaces, converting to lowercase, and handling special characters.
    
    Args:
    data (list): A list of dictionaries, where each dictionary represents a data item.
    
    Returns:
    list: The preprocessed data.
    """
    preprocessed_data = []
    for item in data:
        processed_item = {}
        for key, value in item.items():
            if isinstance(value, str):
                # Clean and preprocess the string value
                processed_value = value.strip().lower()
                # Add more preprocessing steps as needed
                processed_item[key] = processed_value
            else:
                processed_item[key] = value
        preprocessed_data.append(processed_item)
    return preprocessed_data

# Note: Function calls are commented out to prevent execution in this environment.
# Example usage:
# dataset = load_dataset('path/to/your/dataset.json')
# preprocessed_dataset = preprocess_data(dataset)
