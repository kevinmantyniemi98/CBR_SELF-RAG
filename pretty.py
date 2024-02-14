import json
import pprint

def read_and_pretty_print_jsonl(file_path):
    """
    Reads a .jsonl file, parses each JSON object, and
    pretty-prints the content to the console.

    Parameters:
    file_path (str): Path to the .jsonl file to be read.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                # Parse the JSON line
                json_obj = json.loads(line)
                # Pretty-print the JSON object
                pprint.pprint(json_obj, width=80, compact=False)
                print("\n")
            except json.JSONDecodeError:
                print("Error decoding JSON.")

# Replace 'your_file_path.jsonl' with the path to your actual .jsonl file
file_path = 'output_remember_forget2.jsonl'
read_and_pretty_print_jsonl(file_path)