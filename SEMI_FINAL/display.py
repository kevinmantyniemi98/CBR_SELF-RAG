import json
import pprint
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse  # Assuming you're intending to use argparse for command-line arguments

# PrettyPrintJsonl class definition
class PrettyPrintJsonl:
    def __init__(self, filename):
        self.filename = filename
        
    def read_and_pretty_print_jsonl(self):
        with open(self.filename, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    json_obj = json.loads(line)
                    pprint.pprint(json_obj, width=80)
                    print("\n")
                except json.JSONDecodeError:
                    print("Error decoding JSON.")

# JSONLPlotter class definition
class JSONLPlotter:
    def __init__(self, filename, rf_token_column='RF_token'):
        self.filename = filename
        self.rf_token_column = rf_token_column
        self.data = None

    def load_data(self):
        try:
            data = []
            with open(self.filename, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            self.data = pd.DataFrame(data)
        except FileNotFoundError:
            print(f"Error: File not found {self.filename}")

    def plot_rf_token_counts(self):
        if self.data is None:
            print("Data not loaded. Call load_data() first.")
            return

        try:
            counts = self.data[self.rf_token_column].value_counts()
            plt.bar(counts.index.astype(str), counts.values)
            plt.xlabel('RF Token')
            plt.ylabel('Count')
            plt.title(f'Frequency of RF Tokens in {self.filename}')
            plt.show()

        except KeyError:
            print(f"Column '{self.rf_token_column}' not found in the data.")

# Main script
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_filepath = os.path.join(script_dir, 'Self_rag_train.jsonl')
    output_filepath = os.path.join(script_dir, 'output_remember_forget.jsonl')
    
    # Example usage - since the generation function isn't defined, it's skipped here
    file_path = os.path.join('SEMI_FINAL', 'output_remember_forget.jsonl')
    
    # Creating an instance and calling PrettyPrintJsonl
    pp_jsonl = PrettyPrintJsonl(file_path)
    pp_jsonl.read_and_pretty_print_jsonl()
    
    # Creating an instance and calling JSONLPlotter 
    plotter = JSONLPlotter(file_path)
    plotter.load_data()
    plotter.plot_rf_token_counts()