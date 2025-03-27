import time
import random
import pandas as pd

DATA_FILE = "wmc.csv"

def generate_fake_data():
    """Simulates packet traffic and writes to wmc.csv."""
    timestamps = pd.date_range(start="2025-03-27", periods=100, freq="T")
    packets = [random.randint(50, 500) for _ in range(100)]
    
    df = pd.DataFrame({"timestamp": timestamps, "packets": packets})
    df.to_csv(DATA_FILE, index=False)
    print(f"Fake data saved to {DATA_FILE}")

if __name__ == "__main__":
    generate_fake_data()
