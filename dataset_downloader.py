import argparse
import logging
from datasets import load_dataset
import os
import json
from tqdm import tqdm
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Download and Process Hugging Face datasets")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset on Hugging Face.")
    parser.add_argument("--split", type=str, choices=['train', 'validation', 'test'], help="Which split of the dataset to download. If not specified, all splits are downloaded.")
    parser.add_argument("--subset_size", type=int, default=None, help="Size of the dataset subset to download. Download the entire dataset if not specified.")
    return parser.parse_args()

def convert_column_names(dataset, column_mapping):
    """
    Converts column names in a dataset based on a predefined mapping.
    """
    logging.info("Starting to convert column names...")
    for old_name, new_name in column_mapping.items():
        if new_name in dataset.column_names:
            logging.warning(f"Skipping renaming of {old_name} to {new_name} since {new_name} already exists.")
            continue
        if old_name in dataset.column_names:
            logging.debug(f"Renaming column {old_name} to {new_name}")
            dataset = dataset.rename_column(old_name, new_name)
    logging.info("Column name conversion completed.")
    return dataset

def record_dataset_path(path, dataset_paths_file="dataset_paths.pkl"):
    """
    Records a dataset's path in a specified pickle file. Creates the file if it does not exist.
    
    Args:
        path (str): Relative path to the dataset.
        dataset_paths_file (str): Pickle file where dataset paths are recorded.
    """
    try:
        with open(dataset_paths_file, "rb") as file:
            dataset_paths = pickle.load(file)
    except FileNotFoundError:
        dataset_paths = []

    dataset_paths.append(path)

    with open(dataset_paths_file, "wb") as file:
        pickle.dump(dataset_paths, file)

    logging.info(f"Recorded dataset path: {path}")


def load_and_process_dataset(dataset_name, split=None, subset_size=None, save_path="dataset"):
    """
    Loads a dataset from Hugging Face, processes it, and saves it to a specified directory.
    Records each dataset split's path in the dataset_paths.txt file.
    """
    column_name_mapping = {
        #INPUT
        'text': 'instruction',
        'sentence': 'instruction',
        'input': 'instruction',
        'content': 'instruction',
        # OUTPUT
        'label': 'output',
        'target': 'output',
        # Add more mappings as necessary
    }

    if split:
        dataset = load_dataset(dataset_name, split=split)
    else:
        dataset = load_dataset(dataset_name)
    
    if subset_size is not None:
        # Select a subset if subset_size is specified
        dataset = dataset.select(range(subset_size))

    logging.info("Dataset loaded successfully.")

    def process_and_save(current_dataset, current_split):
        logging.info(f"Processing the {current_split} split.")
        processed_split = convert_column_names(current_dataset, column_name_mapping)

        # Adjusted split_save_path to fit the requested directory structure
        if subset_size is not None:
                split_save_path = os.path.join(save_path, dataset_name.replace("/", "_"), f"{current_split}S{str(subset_size)}.json")
        else:
            split_save_path = os.path.join(save_path, dataset_name.replace("/", "_"), f"{current_split}.json")
        os.makedirs(os.path.dirname(split_save_path), exist_ok=True)

        with open(split_save_path, 'w', encoding='utf-8') as f:
            json.dump(processed_split.to_dict(), f, indent=4)
        
        relative_path = os.path.relpath(split_save_path, start=os.curdir)
        record_dataset_path(relative_path)

        logging.info(f"Saved processed {current_split} split to {split_save_path}")

    # Handle whether the loaded dataset is a single split or multiple splits
    if isinstance(dataset, dict):  # Multiple splits
        for current_split in dataset.keys():
            process_and_save(dataset[current_split], current_split)
    else:  # Single split
        process_and_save(dataset, split if split else "full_dataset")

def main():
    args = parse_arguments()
    
    load_and_process_dataset(args.dataset_name, args.split, args.subset_size)

if __name__ == "__main__":
    main()