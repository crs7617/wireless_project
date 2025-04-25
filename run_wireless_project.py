import os
import sys
from visualization import run_self_healing_pipeline

def main():
    """Main function to run the wireless project pipeline"""
    print("="*80)
    print("Wireless Network Self-Healing Project")
    print("="*80)
    
    # Check if required files exist
    required_files = ["wmc.csv", "ai_model.py", "self_heal.py", "visualization.py"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("ERROR: The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease ensure all required files are in the current directory.")
        sys.exit(1)
    
    print("\nAll required files found. Starting the self-healing pipeline...\n")
    
    # Run the self-healing pipeline
    run_self_healing_pipeline()
    
    print("\n")
    print("="*80)
    print("Pipeline completed successfully!")
    print("Results have been saved to the current directory.")
    print("="*80)

if __name__ == "__main__":
    main()