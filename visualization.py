import pandas as pd
import matplotlib.pyplot as plt

def load_and_analyze_data(file_path):
    # Load the data
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.\n")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Display basic information
    print("First few rows of the dataset:")
    print(df.head(), "\n")
    
    # Check timestamp conversion if applicable
    if 'Time' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['Time'], unit='s')
            print("Timestamp conversion check:")
            print(f"Timestamp dtype: {df['timestamp'].dtype}")
            print(f"Number of null timestamps: {df['timestamp'].isnull().sum()}\n")
            
            # Sort by timestamp and show first 10
            df_sorted = df.sort_values('timestamp')
            print("First 10 timestamps after sorting:")
            print(df_sorted['timestamp'].head(10), "\n")
        except Exception as e:
            print(f"Timestamp processing error: {e}\n")
    
    # ====== PLOT 1: Protocol Distribution ======
    if 'Protocol' in df.columns:
        print("\nGenerating Protocol Distribution Plot...")
        protocol_counts = df['Protocol'].value_counts()
        
        plt.figure(figsize=(10, 6))
        protocol_counts.plot(kind='bar', color='skyblue')
        plt.title('Protocol Distribution')
        plt.xlabel('Protocol')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        print("Protocol distribution counts:")
        print(protocol_counts, "\n")
    
    # ====== PLOT 2: Top IPs ======
    if 'Source' in df.columns and 'Destination' in df.columns:
        print("\nGenerating Top IPs Plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Top 5 Source IPs
        top_sources = df['Source'].value_counts().head(5)
        top_sources.plot(kind='bar', ax=ax1, color='lightgreen')
        ax1.set_title('Top 5 Source IPs')
        ax1.set_xlabel('IP Address')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Top 5 Destination IPs
        top_dests = df['Destination'].value_counts().head(5)
        top_dests.plot(kind='bar', ax=ax2, color='salmon')
        ax2.set_title('Top 5 Destination IPs')
        ax2.set_xlabel('IP Address')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        print("Top 5 Source IPs:")
        print(top_sources, "\n")
        print("Top 5 Destination IPs:")
        print(top_dests, "\n")
    
    print("\nAnalysis complete. Only Protocol Distribution and Top IPs plots were generated.")

# Example usage
if __name__ == "__main__":
    file_path = "wmc.csv"  # Replace with your actual file path
    load_and_analyze_data(file_path)