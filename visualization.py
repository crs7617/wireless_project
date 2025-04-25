import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from matplotlib.patches import Patch
import seaborn as sns

def load_and_analyze_data(original_file_path, healed_file_path=None):
    """
    Load and visualize network data with self-healing comparison
    Args:
        original_file_path: Path to the original CSV data
        healed_file_path: Path to the healed CSV data (if available)
    """
    # Load the original data
    try:
        df_original = pd.read_csv(original_file_path)
        print(f"Original data loaded successfully from {original_file_path}.\n")
        print(f"Dataset has {len(df_original)} records and {len(df_original.columns)} columns.\n")
        
        # Check if healed data exists
        has_healed_data = False
        if healed_file_path and os.path.exists(healed_file_path):
            df_healed = pd.read_csv(healed_file_path)
            print(f"Healed data loaded successfully from {healed_file_path}.\n")
            has_healed_data = True
        else:
            print("No healed data provided or file not found.\n")
            if healed_file_path:
                print(f"Could not find file: {healed_file_path}\n")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Display basic information
    print("First few rows of the original dataset:")
    print(df_original.head(), "\n")
    
    # Check timestamp conversion if applicable
    if 'Time' in df_original.columns:
        try:
            df_original['timestamp'] = pd.to_datetime(df_original['Time'], unit='s')
            print("Timestamp conversion check:")
            print(f"Timestamp dtype: {df_original['timestamp'].dtype}")
            print(f"Number of null timestamps: {df_original['timestamp'].isnull().sum()}\n")
            
            # Sort by timestamp and show first 10
            df_sorted = df_original.sort_values('timestamp')
            print("First 10 timestamps after sorting:")
            print(df_sorted['timestamp'].head(10), "\n")
        except Exception as e:
            print(f"Timestamp processing error: {e}\n")
    
    # ====== PLOT 1: Protocol Distribution ======
    if 'Protocol' in df_original.columns:
        print("\nGenerating Protocol Distribution Plot...")
        protocol_counts = df_original['Protocol'].value_counts()
        
        plt.figure(figsize=(10, 6))
        protocol_counts.plot(kind='bar', color='skyblue')
        plt.title('Protocol Distribution')
        plt.xlabel('Protocol')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('protocol_distribution.png')
        plt.show()
        
        print("Protocol distribution counts:")
        print(protocol_counts, "\n")
    
    # ====== PLOT 2: Top IPs ======
    if 'Source' in df_original.columns and 'Destination' in df_original.columns:
        print("\nGenerating Top IPs Plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Top 5 Source IPs
        top_sources = df_original['Source'].value_counts().head(5)
        top_sources.plot(kind='bar', ax=ax1, color='lightgreen')
        ax1.set_title('Top 5 Source IPs')
        ax1.set_xlabel('IP Address')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Top 5 Destination IPs
        top_dests = df_original['Destination'].value_counts().head(5)
        top_dests.plot(kind='bar', ax=ax2, color='salmon')
        ax2.set_title('Top 5 Destination IPs')
        ax2.set_xlabel('IP Address')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('top_ips.png')
        plt.show()
        
        print("Top 5 Source IPs:")
        print(top_sources, "\n")
        print("Top 5 Destination IPs:")
        print(top_dests, "\n")
    
    # ====== NEW ADDITION: Network Health Visualization ======
    metrics = ['signal_strength', 'packet_loss', 'latency', 'throughput']
    available_metrics = [m for m in metrics if m in df_original.columns]
    
    if available_metrics:
        print("\nGenerating Network Health Metrics Visualization...")
        
        # If we have anomaly data, we can highlight the anomalies
        anomaly_indices = []
        if 'is_anomaly' in df_original.columns:
            anomaly_indices = df_original[df_original['is_anomaly'] == 1].index.tolist()
        
        # Plot network health metrics - Before healing
        visualize_network_health(df_original, available_metrics, anomaly_indices, 
                                'Network Health Metrics (Before Healing)', 
                                'network_health_before.png')
        
        # If we have healed data, plot comparison
        if has_healed_data:
            # Plot network health metrics - After healing
            visualize_network_health(df_healed, available_metrics, [],
                                    'Network Health Metrics (After Healing)',
                                    'network_health_after.png')
            
            # Plot before-after comparison
            visualize_healing_comparison(df_original, df_healed, available_metrics,
                                        'network_healing_comparison.png')
            
            # Plot healing impact
            visualize_healing_impact(df_original, df_healed, available_metrics,
                                    'healing_impact.png')
    
    print("\nAnalysis complete. All visualizations have been generated and saved.")

def visualize_network_health(df, metrics, anomaly_indices=None, title="Network Health Metrics", save_path=None):
    """Visualize network health metrics with anomaly highlighting"""
    normal_color = "#50C878"  # Emerald green
    anomaly_color = "#FF5050"  # Light red
    
    # Determine grid layout
    n_plots = len(metrics)
    if n_plots <= 2:
        fig, axes = plt.subplots(1, n_plots, figsize=(12, 5))
    else:
        fig, axes = plt.subplots((n_plots+1)//2, 2, figsize=(14, 10))
    
    # Make sure axes is always iterable
    if n_plots == 1:
        axes = [axes]
    elif n_plots > 2:
        axes = axes.flatten()
    
    fig.suptitle(title, fontsize=16)
    
    for i, metric in enumerate(metrics):
        if i < len(axes):
            ax = axes[i]
            
            # Create colors array if anomaly indices provided
            if anomaly_indices:
                colors = np.array([normal_color] * len(df))
                colors[anomaly_indices] = anomaly_color
            else:
                colors = normal_color
            
            # Plot data points
            ax.scatter(range(len(df)), df[metric], c=colors, alpha=0.7, s=40)
            
            # Set appropriate title and labels
            metric_label = ' '.join(word.capitalize() for word in metric.split('_'))
            ax.set_title(metric_label)
            ax.set_xlabel("Data Point")
            
            # Set y-label and thresholds based on metric
            if metric == 'signal_strength':
                ax.set_ylabel("Signal Strength (dBm)")
                ax.set_ylim([-100, -30])
                ax.axhline(y=-80, color='r', linestyle='--', alpha=0.5, label="Poor Signal Threshold")
            elif metric == 'packet_loss':
                ax.set_ylabel("Packet Loss (%)")
                ax.set_ylim([0, 25])
                ax.axhline(y=5, color='r', linestyle='--', alpha=0.5, label="High Loss Threshold")
            elif metric == 'latency':
                ax.set_ylabel("Latency (ms)")
                ax.set_ylim([0, 250])
                ax.axhline(y=100, color='r', linestyle='--', alpha=0.5, label="High Latency Threshold")
            elif metric == 'throughput':
                ax.set_ylabel("Throughput (Mbps)")
                ax.set_ylim([0, 50])
                ax.axhline(y=10, color='r', linestyle='--', alpha=0.5, label="Low Throughput Threshold")
            
            ax.legend()
    
    # Hide any unused subplots
    for i in range(len(metrics), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Network health visualization saved to {save_path}")
    
    plt.show()

def visualize_healing_comparison(df_original, df_healed, metrics, save_path=None):
    """Visualize before and after healing comparison for network metrics"""
    normal_color = "#50C878"    # Emerald green
    anomaly_color = "#FF5050"   # Light red
    healed_color = "#4169E1"    # Royal blue
    
    # Create figure with 2 rows
    fig, axes = plt.subplots(len(metrics), 2, figsize=(16, 4*len(metrics)))
    if len(metrics) == 1:
        axes = axes.reshape(1, 2)
    
    fig.suptitle("Network Health: Before vs After Self-Healing", fontsize=18, y=0.98)
    
    # Create legend elements
    legend_elements = [
        Patch(facecolor=normal_color, label='Normal'),
        Patch(facecolor=anomaly_color, label='Anomaly (Before)'),
        Patch(facecolor=healed_color, label='After Healing')
    ]
    
    # Function to determine threshold based on metric
    def get_threshold_info(metric):
        if metric == 'signal_strength':
            return -80, "Poor Signal Threshold"
        elif metric == 'packet_loss':
            return 5, "High Loss Threshold"
        elif metric == 'latency':
            return 100, "High Latency Threshold"
        elif metric == 'throughput':
            return 10, "Low Throughput Threshold"
        return None, None
    
    # Function to set appropriate y-limits
    def get_ylim(metric):
        if metric == 'signal_strength':
            return [-100, -30]
        elif metric == 'packet_loss':
            return [0, 25]
        elif metric == 'latency':
            return [0, 250]
        elif metric == 'throughput':
            return [0, 50]
        return None
    
    # Identify anomalies in original data 
    # Simplified approach - could be replaced with actual anomaly detection
    anomaly_indices = {
        'signal_strength': df_original[df_original['signal_strength'] < -80].index,
        'packet_loss': df_original[df_original['packet_loss'] > 5].index,
        'latency': df_original[df_original['latency'] > 100].index,
        'throughput': df_original[df_original['throughput'] < 10].index
    }
    
    for i, metric in enumerate(metrics):
        # Set friendly metric name
        metric_name = ' '.join(word.capitalize() for word in metric.split('_'))
        
        # Before healing subplot
        ax1 = axes[i, 0]
        
        # Create colors for anomaly highlighting
        colors = np.array([normal_color] * len(df_original))
        if metric in anomaly_indices:
            colors[anomaly_indices[metric]] = anomaly_color
        
        # Plot original data
        ax1.scatter(range(len(df_original)), df_original[metric], c=colors, alpha=0.7, s=30)
        ax1.set_title(f"Before Healing: {metric_name}")
        ax1.set_ylabel(metric_name)
        
        # After healing subplot
        ax2 = axes[i, 1]
        ax2.scatter(range(len(df_healed)), df_healed[metric], c=healed_color, alpha=0.7, s=30)
        ax2.set_title(f"After Healing: {metric_name}")
        ax2.set_ylabel(metric_name)
        
        # Set y-limits and add threshold lines
        threshold, label = get_threshold_info(metric)
        ylim = get_ylim(metric)
        
        if ylim:
            ax1.set_ylim(ylim)
            ax2.set_ylim(ylim)
        
        if threshold is not None:
            ax1.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label=label)
            ax2.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label=label)
    
    # Add a common legend at the bottom
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Healing comparison visualization saved to {save_path}")
    
    plt.show()

def visualize_healing_impact(df_original, df_healed, metrics, save_path=None):
    """Visualize the impact of healing on network metrics using bar charts"""
    original_color = "#FF9966"  # Light orange
    healed_color = "#66B2FF"    # Light blue
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    if len(metrics) <= 2:
        axes = axes[:1, :]
    
    axes = axes.flatten()
    fig.suptitle("Impact of Self-Healing on Network Metrics", fontsize=16)
    
    for i, metric in enumerate(metrics):
        if i < len(axes):
            ax = axes[i]
            
            # Compute statistics
            before_mean = df_original[metric].mean()
            after_mean = df_healed[metric].mean()
            
            # Determine if improvement is higher or lower value
            if metric in ['signal_strength', 'throughput']:
                # Higher is better
                improvement = (after_mean - before_mean) / abs(before_mean) * 100
                better = after_mean > before_mean
            else:
                # Lower is better
                improvement = (before_mean - after_mean) / before_mean * 100
                better = after_mean < before_mean
            
            # Create bar chart
            bars = ax.bar(
                ['Before Healing', 'After Healing'],
                [before_mean, after_mean],
                color=[original_color, healed_color]
            )
            
            # Add percentage improvement text
            if better:
                improvement_text = f"+{improvement:.1f}% Improvement"
                color = 'green'
            else:
                improvement_text = f"{improvement:.1f}% Change"
                color = 'red'
                
            ax.text(0.5, 0.9, improvement_text, 
                   transform=ax.transAxes, ha='center', 
                   fontsize=12, color=color, weight='bold')
            
            # Set appropriate labels
            metric_name = ' '.join(word.capitalize() for word in metric.split('_'))
            ax.set_title(metric_name)
            ax.set_ylabel(f"Average {metric_name}")
            
            # Add data labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f"{height:.2f}", ha='center', va='bottom')
    
    # Hide any unused subplots
    for i in range(len(metrics), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Healing impact visualization saved to {save_path}")
    
    plt.show()

# Main function to run the entire pipeline
def run_self_healing_pipeline():
    """Run the complete self-healing pipeline and visualization"""
    from ai_model import AnomalyDetector
    from self_heal import SelfHealingMechanism
    import os
    
    print("="*60)
    print("Starting Wireless Network Self-Healing Pipeline")
    print("="*60)
    
    # Define file paths
    original_data_path = "wmc.csv"
    healed_data_path = "wmc_healed.csv"
    
    # Step 1: Check for anomalies
    print("\nStep 1: Detecting Anomalies...")
    detector = AnomalyDetector()
    
    # Load data
    try:
        data = pd.read_csv(original_data_path)
        print(f"Loaded data with {len(data)} records")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Train detector if not already trained
    detector.train(data)
    
    # Detect anomalies
    anomaly_indices, anomaly_scores, anomaly_details = detector.detect_anomalies(data)
    
    print(f"Detected {len(anomaly_indices)} anomalies in the data")
    for i, detail in enumerate(anomaly_details[:5]):  # Show first 5 anomalies
        print(f"  Anomaly {i+1}: {detail['type']}, Severity: {detail['severity']}")
    
    # Mark anomalies in the dataset for visualization
    data['is_anomaly'] = 0
    data.loc[anomaly_indices, 'is_anomaly'] = 1
    
    # Step 2: Visualize original data with anomalies
    print("\nStep 2: Visualizing Original Network State with Anomalies...")
    visualize_network_health(data, ['signal_strength', 'packet_loss', 'latency', 'throughput'], 
                           anomaly_indices, "Network Health with Detected Anomalies", 
                           "anomalies_detected.png")
    
    # Step 3: Apply self-healing
    print("\nStep 3: Applying Self-Healing Mechanism...")
    healer = SelfHealingMechanism(data_path=original_data_path)
    healed_data, healing_actions = healer.perform_self_healing()
    
    print(f"Applied {len(healing_actions)} healing actions")
    for i, action in enumerate(healing_actions[:5]):  # Show first 5 healing actions
        print(f"  Action {i+1}: {action['type']} - {action['action']}")
        print(f"    Before: {action['before']:.2f}, After: {action['after']:.2f}")
    
    # Generate healing report
    report_path = healer.generate_healing_report()
    print(f"Healing report generated at {report_path}")
    
    # Step 4: Visualize before and after healing
    print("\nStep 4: Visualizing Before vs After Healing...")
    
    # Ensure healed data exists
    if os.path.exists(healed_data_path):
        load_and_analyze_data(original_data_path, healed_data_path)
    else:
        print(f"Healed data file not found at {healed_data_path}")
        # Use the healed_data object instead
        healed_data.to_csv(healed_data_path, index=False)
        print(f"Created healed data file at {healed_data_path}")
        load_and_analyze_data(original_data_path, healed_data_path)
    
    print("\nSelf-Healing Pipeline Complete!")
    print("="*60)

# Example usage
if __name__ == "__main__":
    # Option 1: Just visualize existing data
    # load_and_analyze_data("wmc.csv", "wmc_healed.csv")
    
    # Option 2: Run complete self-healing pipeline
    run_self_healing_pipeline()