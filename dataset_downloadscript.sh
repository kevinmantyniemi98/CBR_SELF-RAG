#!/bin/bash

# Script to run the Python file for processing the "selfrag/selfrag_train_data" train split

# Assuming your Python script is named process_dataset.py
# Replace /path/to/your/python/script with the actual path to the Python file

PYTHON_SCRIPT="/path/to/your/python/script/process_dataset.py"

# Activate your Python environment if necessary
# source /path/to/your/env/bin/activate

# Run the script for the selfrag/selfrag_train_data dataset, train split
python $PYTHON_SCRIPT --dataset_name "selfrag/selfrag_train_data" --split "train"

echo "Dataset processing completed."
