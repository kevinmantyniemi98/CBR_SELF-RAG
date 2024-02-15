import pandas as pd
import json
import matplotlib.pyplot as plt

class JSONLPlotter:
    def __init__(self, filename, rf_token_column='RF_token'):
        """
        Initializes the JSONLPlotter class.

        Args:
            filename (str): Path to the JSONL file.
            rf_token_column (str, optional): The column name containing the RF tokens. 
                                         Defaults to 'RF_token'.
        """
        self.filename = filename
        self.rf_token_column = rf_token_column
        self.data = None

    def load_data(self):
        """Loads data from the JSONL file."""
        try:
            data = []
            with open(self.filename, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            self.data = pd.DataFrame(data)
        except FileNotFoundError:
            print(f"Error: File not found {self.filename}")

    def plot_rf_token_counts(self):
        """Calculates and plots the frequency of RF tokens."""
        if self.data is None:
            print("Error: Data not loaded. Call load_data() first.")
            return

        try:
            counts = self.data[self.rf_token_column].value_counts()
            labels = counts.index.astype(str).to_list()  # Convert index to strings
            
            plt.bar(labels, counts)
            plt.xlabel('RF Token')
            plt.ylabel('Count')
            plt.title(f'Frequency of RF Tokens in {self.filename}') 
            plt.show()

        except KeyError:
            print(f"Error: Column '{self.rf_token_column}' not found in the data.")

if __name__ == "__main__":
    plotter = JSONLPlotter('SEMI_FINAL\output_remember_forget.jsonl')  
    plotter.load_data()
    plotter.plot_rf_token_counts()
