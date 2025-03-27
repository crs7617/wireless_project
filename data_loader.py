import os
import pandas as pd

# Define file path
DATA_PATH = os.path.join(os.path.dirname(__file__), "wmc.csv")

def load_data():
    """Loads wmc.csv into a DataFrame."""
    try:
        df = pd.read_csv(DATA_PATH)
        print("Data loaded successfully!")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
