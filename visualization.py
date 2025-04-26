import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from datetime import datetime, timedelta
import os
import random
from ai_model import AnomalyDetector
from self_heal import SelfHealingSystem
import time

# Create output directories if they don't exist
os.makedirs('output/reports', exist_ok=True)
os.makedirs('output/visualizations', exist_ok=True)

# Helper functions for visualization
def save_plot(fig, filename):
    """Save the figure to the visualizations directory."""
    filepath = os.path.join('output/visualizations', filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return filepath

def generate_heatmap(data, metric, title, filename):
    """Generate a heatmap for a specific metric."""
    if 'position_x' not in data.columns or 'position_y' not in data.columns:
        print(f"Cannot create heatmap: position_x or position_y columns not found")
        return None
    
    # Check if the metric exists in the data
    if metric not in data.columns:
        print(f"Cannot create heatmap: {metric} column not found")
        return None
    
    # Create a pivot table for the heatmap
    pivot_data = data.pivot_table(
        values=metric,
        index='position_y',
        columns='position_x',
        aggfunc='mean'
    )
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot_data, cmap='viridis', annot=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    
    # Save the plot
    return save_plot(fig, filename)

def plot_metric_over_time(data, metric, title, filename):
    """Plot a metric's values over time."""
    if 'timestamp' not in data.columns:
        print(f"Cannot create time plot: timestamp column not found")
        return None
    
    # Check if the metric exists in the data
    if metric not in data.columns:
        print(f"Cannot create time plot: {metric} column not found")
        return None
    
    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['timestamp'], data[metric], marker='o', linestyle='-', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    return save_plot(fig, filename)

def visualize_anomalies(data, anomalies, scores):
    """Visualize the detected anomalies."""
    if len(data) == 0:
        print("No data to visualize")
        return None
    
    # Create a copy of the data to avoid modifying the original
    viz_data = data.copy()
    viz_data['anomaly'] = anomalies
    viz_data['anomaly_score'] = scores
    
    # Get numeric columns for potential visualization
    numeric_cols = viz_data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove the anomaly columns from numeric_cols
    numeric_cols = [col for col in numeric_cols if col not in ['anomaly', 'anomaly_score']]
    
    # Check if position columns exist
    has_position = all(col in viz_data.columns for col in ['position_x', 'position_y'])
    
    # Create visualizations
    plots = []
    
    # 1. Anomaly Score Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(viz_data['anomaly_score'], kde=True, ax=ax)
    ax.set_title('Distribution of Anomaly Scores')
    ax.set_xlabel('Anomaly Score (lower = more anomalous)')
    ax.axvline(x=viz_data[viz_data['anomaly']]['anomaly_score'].max(), 
               color='red', linestyle='--', label='Anomaly Threshold')
    ax.legend()
    plots.append(save_plot(fig, 'anomaly_score_distribution.png'))
    
    # 2. Spatial Distribution of Anomalies (if position data exists)
    if has_position:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            viz_data['position_x'], 
            viz_data['position_y'],
            c=viz_data['anomaly_score'],
            cmap='coolwarm_r',
            alpha=0.7,
            s=50
        )
        ax.set_title('Spatial Distribution of Anomalies')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        plt.colorbar(scatter, label='Anomaly Score (lower = more anomalous)')
        
        # Highlight anomalies with red circles
        anomaly_data = viz_data[viz_data['anomaly']]
        if len(anomaly_data) > 0:
            ax.scatter(
                anomaly_data['position_x'],
                anomaly_data['position_y'],
                facecolors='none',
                edgecolors='red',
                s=100,
                linewidths=2,
                label='Anomalies'
            )
            ax.legend()
        
        plots.append(save_plot(fig, 'anomaly_spatial_distribution.png'))
    
    # 3. For each numeric column, create a scatter plot with anomaly scores
    for col in numeric_cols[:5]:  # Limit to first 5 columns for brevity
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            viz_data[col],
            viz_data['anomaly_score'],
            c=viz_data['anomaly_score'],
            cmap='coolwarm_r',
            alpha=0.7
        )
        ax.set_title(f'Anomaly Score vs {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Anomaly Score')
        plt.colorbar(scatter, label='Anomaly Score (lower = more anomalous)')
        
        # Highlight anomalies
        anomaly_data = viz_data[viz_data['anomaly']]
        if len(anomaly_data) > 0:
            ax.scatter(
                anomaly_data[col],
                anomaly_data['anomaly_score'],
                facecolors='none',
                edgecolors='red',
                s=100,
                linewidths=2,
                label='Anomalies'
            )
            ax.legend()
        
        plots.append(save_plot(fig, f'anomaly_vs_{col}.png'))
    
    # 4. If timestamp exists, plot anomalies over time
    if 'timestamp' in viz_data.columns:
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(viz_data['timestamp']):
            viz_data['timestamp'] = pd.to_datetime(viz_data['timestamp'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(
            viz_data['timestamp'],
            viz_data['anomaly_score'],
            c=viz_data['anomaly_score'],
            cmap='coolwarm_r',
            alpha=0.7
        )
        ax.set_title('Anomalies Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Anomaly Score')
        plt.xticks(rotation=45)
        
        # Highlight anomalies
        anomaly_data = viz_data[viz_data['anomaly']]
        if len(anomaly_data) > 0:
            ax.scatter(
                anomaly_data['timestamp'],
                anomaly_data['anomaly_score'],
                facecolors='none',
                edgecolors='red',
                s=100,
                linewidths=2,
                label='Anomalies'
            )
            ax.legend()
        
        plt.tight_layout()
        plots.append(save_plot(fig, 'anomalies_over_time.png'))
    
    return plots

def generate_network_dashboard(data):
    """Generate a comprehensive network performance dashboard."""
    metrics = {
        'signal_strength': 'Signal Strength (dBm)',
        'packet_loss': 'Packet Loss (%)',
        'latency': 'Latency (ms)',
        'throughput': 'Throughput (Mbps)'
    }
    
    # Check which metrics exist in the data
    available_metrics = [m for m in metrics if m in data.columns]
    
    if not available_metrics:
        print("No network metrics found in the data")
        return None
    
    # Create a dashboard with subplots
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))
    
    # If only one metric, axes isn't an array
    if n_metrics == 1:
        axes = [axes]
    
    # Convert timestamp to datetime if it exists
    if 'timestamp' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Plot each metric
    for i, metric in enumerate(available_metrics):
        if 'timestamp' in data.columns:
            axes[i].plot(data['timestamp'], data[metric], marker='o', linestyle='-', alpha=0.7)
            axes[i].set_xlabel('Time')
        else:
            axes[i].plot(data.index, data[metric], marker='o', linestyle='-', alpha=0.7)
            axes[i].set_xlabel('Data Point')
            
        axes[i].set_ylabel(metrics[metric])
        axes[i].set_title(f'{metrics[metric]} Over Time')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return save_plot(fig, 'network_dashboard.png')

def generate_optimization_report(before_data, after_data, changes):
    """Generate a report comparing network performance before and after optimization."""
    if len(before_data) == 0 or len(after_data) == 0:
        print("No data for optimization report")
        return None
    
    # Create the reports directory if it doesn't exist
    os.makedirs('output/reports', exist_ok=True)
    
    # Generate a text report
    report_text = "Network Optimization Report\n"
    report_text += "=" * 30 + "\n\n"
    report_text += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report_text += "Changes Applied:\n"
    for change in changes:
        report_text += f"- {change}\n"
    
    # Define the metrics we care about
    metrics = {
        'signal_strength': {'title': 'Signal Strength', 'unit': 'dBm', 'better': 'higher'},
        'packet_loss': {'title': 'Packet Loss', 'unit': '%', 'better': 'lower'},
        'latency': {'title': 'Latency', 'unit': 'ms', 'better': 'lower'},
        'throughput': {'title': 'Throughput', 'unit': 'Mbps', 'better': 'higher'}
    }
    
    # Check which metrics exist in the data
    available_metrics = [m for m in metrics if m in before_data.columns and m in after_data.columns]
    
    report_text += "\nPerformance Metrics:\n"
    for metric in available_metrics:
        before_mean = before_data[metric].mean()
        after_mean = after_data[metric].mean()
        
        if metrics[metric]['better'] == 'higher':
            change_pct = ((after_mean - before_mean) / abs(before_mean)) * 100 if before_mean != 0 else 0
            change_str = "increased" if change_pct > 0 else "decreased"
        else:
            change_pct = ((before_mean - after_mean) / abs(before_mean)) * 100 if before_mean != 0 else 0
            change_str = "decreased" if change_pct > 0 else "increased"
        
        report_text += f"- {metrics[metric]['title']}: {change_str} by {abs(change_pct):.2f}% "
        report_text += f"(Before: {before_mean:.2f} {metrics[metric]['unit']}, After: {after_mean:.2f} {metrics[metric]['unit']})\n"
    
    # Save the report to a file
    report_file = os.path.join('output/reports', 'optimization_report.txt')
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print(f"Report saved to {report_file}")
    return report_file

def run_self_healing_pipeline():
    """Run the complete self-healing pipeline."""
    start_time = time.time()
    
    print("Step 1: Detecting Anomalies...")
    # Load the data
    try:
        data = pd.read_csv('wmc.csv')
        print(f"Loaded data with {len(data)} records")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Initialize the anomaly detector
    detector = AnomalyDetector(contamination=0.05)
    
    # Train the model
    detector.train(data)
    
    # Detect anomalies
    anomalies, scores = detector.detect_anomalies(data)
    anomaly_count = sum(anomalies)
    print(f"Detected {anomaly_count} anomalies ({anomaly_count/len(data)*100:.2f}% of data)")
    
    # Visualize the anomalies
    print("\nStep 2: Visualizing the anomalies...")
    viz_paths = visualize_anomalies(data, anomalies, scores)
    if viz_paths:
        print(f"Generated {len(viz_paths)} visualizations in the output/visualizations directory")
    
    # Get detailed information about the anomalies
    anomaly_details = detector.get_anomaly_details(data, anomalies, scores)
    
    print("\nStep 3: Applying self-healing measures...")
    healer = SelfHealingSystem()
    
    # Get the anomalous data points
    anomaly_data = data.iloc[np.where(anomalies)[0]]
    
    # Apply healing strategies
    healing_plan = healer.generate_healing_plan(anomaly_data)
    print("\nHealing plan generated:")
    for strategy in healing_plan:
        print(f"- {strategy['description']}")
    
    # Simulate the application of the healing strategies
    optimized_data, changes = healer.apply_healing_strategies(data, healing_plan)
    
    # In visualization.py, find the section where it reads the report file
    print("\nStep 4: Generating optimization report...")
    report_path = generate_optimization_report(data, optimized_data, changes)

    if report_path:
        print(f"Generated optimization report at {report_path}")
        
        # Read the report for printing
        try:
            with open(report_path, 'r') as f:
                report_content = f.read()
            
            print("\nOptimization Report Summary:")
            print("-" * 30)
            # Print first 15 lines of the report
            for line in report_content.split('\n')[:15]:
                print(line)
            print("...")
        except Exception as e:
            print(f"Warning: Could not read report file: {e}")

        # Generate a network dashboard
        print("\nStep 5: Generating network performance dashboard...")
        dashboard_path = generate_network_dashboard(optimized_data)
        if dashboard_path:
            print(f"Generated network dashboard at {dashboard_path}")
        
        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nSelf-healing pipeline completed in {execution_time:.2f} seconds")

if __name__ == "__main__":
    run_self_healing_pipeline()
