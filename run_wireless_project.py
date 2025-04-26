import os
import pandas as pd
from visualization import run_self_healing_pipeline

def main():
    # Check if the data file exists
    if not os.path.exists('wmc.csv'):
        print("Error: wmc.csv file not found. Please ensure it's in the correct location.")
        return
    
    # Print information about the dataset
    try:
        data = pd.read_csv('wmc.csv')
        print(f"Dataset Information:")
        print(f"- Number of records: {len(data)}")
        print(f"- Columns available: {', '.join(data.columns.tolist())}")
        print(f"- Time range: {data['timestamp'].min()} to {data['timestamp'].max()}" if 'timestamp' in data.columns else "- No timestamp column found")
        print("\nStarting self-healing pipeline...\n")
    except Exception as e:
        print(f"Error reading data file: {e}")
        return
    
    # Run the self-healing pipeline
    run_self_healing_pipeline()
    
    print("\nWireless Network Self-Healing Project completed!")
    print("Check the 'output' directory for visualizations and reports.")

if __name__ == "__main__":
    main()