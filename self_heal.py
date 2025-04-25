import pandas as pd
import numpy as np
import time
import os
import logging
from ai_model import AnomalyDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='self_heal.log'
)
logger = logging.getLogger('self_heal')

class NetworkOptimizer:
    """Class to optimize network parameters based on detected anomalies."""
    
    def __init__(self):
        self.optimization_history = []
    
    def optimize_channel(self, current_channel):
        """Determine optimal channel with less interference."""
        # Simulated logic to find a better channel
        channels = [1, 6, 11]  # Common non-overlapping channels
        if current_channel in channels:
            channels.remove(current_channel)
        return np.random.choice(channels)
    
    def optimize_power(self, current_power):
        """Adjust transmit power for better signal quality."""
        # Increase power if it's too low, with an upper limit
        if current_power < 15:
            return min(current_power + 3, 20)  # Increase by 3, max 20
        return current_power
    
    def optimize_position(self, current_position):
        """Suggest optimal position for device/AP."""
        # Simulate position optimization with slight adjustments
        x, y, z = current_position
        # Add small adjustments (could be based on signal propagation models)
        new_x = x + np.random.uniform(-1, 1)
        new_y = y + np.random.uniform(-1, 1)
        new_z = z
        return (new_x, new_y, new_z)

class SelfHealingMechanism:
    """Main class implementing the self-healing functionality."""
    
    def __init__(self, data_path="wmc.csv", model_path=None):
        self.data_path = data_path
        self.data = None
        self.anomaly_detector = AnomalyDetector(model_path)
        self.network_optimizer = NetworkOptimizer()
        self.healing_history = []
        self.load_data()
        
    def load_data(self):
        """Load network data for analysis."""
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Loaded data from {self.data_path}: {len(self.data)} records")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.data = None
            
    def train_detector(self):
        """Train the anomaly detector if needed."""
        if self.data is not None and not self.anomaly_detector.is_trained:
            logger.info("Training anomaly detector...")
            self.anomaly_detector.train(self.data)
            logger.info("Anomaly detector trained successfully")
    
    def detect_anomalies(self):
        """Detect anomalies in the network data."""
        if self.data is None:
            logger.error("No data available for anomaly detection")
            return [], [], []
            
        if not self.anomaly_detector.is_trained:
            self.train_detector()
            
        logger.info("Detecting anomalies...")
        anomaly_indices, anomaly_scores, anomaly_details = self.anomaly_detector.detect_anomalies(self.data)
        logger.info(f"Detected {len(anomaly_indices)} anomalies")
        return anomaly_indices, anomaly_scores, anomaly_details
    
    def apply_healing_actions(self, anomaly_details):
        """Apply healing actions based on detected anomalies."""
        healed_data = self.data.copy()
        healing_actions = []
        
        for anomaly in anomaly_details:
            idx = anomaly['index']
            anomaly_type = anomaly['type']
            severity = anomaly['severity']
            
            # Record pre-healing state
            pre_healing = healed_data.iloc[idx].copy()
            
            # Apply healing based on anomaly type
            if anomaly_type == "weak_signal":
                # Optimize channel and power
                current_channel = healed_data.loc[idx, 'channel'] if 'channel' in healed_data.columns else 1
                new_channel = self.network_optimizer.optimize_channel(current_channel)
                
                current_power = healed_data.loc[idx, 'transmit_power'] if 'transmit_power' in healed_data.columns else 10
                new_power = self.network_optimizer.optimize_power(current_power)
                
                # Update values
                if 'channel' in healed_data.columns:
                    healed_data.loc[idx, 'channel'] = new_channel
                if 'transmit_power' in healed_data.columns:
                    healed_data.loc[idx, 'transmit_power'] = new_power
                
                # Simulate improvement in signal strength
                healed_data.loc[idx, 'signal_strength'] = min(-55, healed_data.loc[idx, 'signal_strength'] + 20)
                
                healing_actions.append({
                    'index': idx,
                    'type': 'weak_signal',
                    'action': f"Changed channel to {new_channel}, increased power to {new_power}",
                    'before': pre_healing['signal_strength'],
                    'after': healed_data.loc[idx, 'signal_strength']
                })
                
            elif anomaly_type == "high_packet_loss":
                # Reduce interference and improve reliability
                current_channel = healed_data.loc[idx, 'channel'] if 'channel' in healed_data.columns else 1
                new_channel = self.network_optimizer.optimize_channel(current_channel)
                
                if 'channel' in healed_data.columns:
                    healed_data.loc[idx, 'channel'] = new_channel
                
                # Simulate improvement in packet loss
                healed_data.loc[idx, 'packet_loss'] = max(0.5, healed_data.loc[idx, 'packet_loss'] * 0.3)
                
                healing_actions.append({
                    'index': idx,
                    'type': 'high_packet_loss',
                    'action': f"Changed channel to {new_channel} to reduce interference",
                    'before': pre_healing['packet_loss'],
                    'after': healed_data.loc[idx, 'packet_loss']
                })
                
            elif anomaly_type == "high_latency":
                # Optimize routing and QoS
                healed_data.loc[idx, 'latency'] = max(20, healed_data.loc[idx, 'latency'] * 0.6)
                
                healing_actions.append({
                    'index': idx,
                    'type': 'high_latency',
                    'action': "Optimized routing and applied QoS settings",
                    'before': pre_healing['latency'],
                    'after': healed_data.loc[idx, 'latency']
                })
                
            elif anomaly_type == "low_throughput":
                # Improve channel width and modulation
                healed_data.loc[idx, 'throughput'] = min(50, healed_data.loc[idx, 'throughput'] * 2.5)
                
                healing_actions.append({
                    'index': idx,
                    'type': 'low_throughput',
                    'action': "Increased channel width and optimized modulation scheme",
                    'before': pre_healing['throughput'],
                    'after': healed_data.loc[idx, 'throughput']
                })
        
        self.healing_history = healing_actions
        return healed_data, healing_actions
    
    def perform_self_healing(self):
        """Main method to perform the self-healing process."""
        logger.info("Starting self-healing process...")
        
        # Step 1: Detect anomalies
        anomaly_indices, anomaly_scores, anomaly_details = self.detect_anomalies()
        if not anomaly_details:
            logger.info("No anomalies detected, network is healthy")
            return self.data, []
        
        # Step 2: Apply healing actions
        healed_data, healing_actions = self.apply_healing_actions(anomaly_details)
        
        # Step 3: Save healed data
        healed_path = self.data_path.replace('.csv', '_healed.csv')
        healed_data.to_csv(healed_path, index=False)
        logger.info(f"Healed data saved to {healed_path}")
        
        # Step 4: Log healing results
        self._log_healing_results(healing_actions)
        
        return healed_data, healing_actions
    
    def _log_healing_results(self, healing_actions):
        """Log the results of healing actions."""
        if not healing_actions:
            logger.info("No healing actions were performed")
            return
            
        logger.info(f"Applied {len(healing_actions)} healing actions:")
        for action in healing_actions:
            logger.info(f"- {action['type']}: {action['action']} (Before: {action['before']}, After: {action['after']})")
    
    def generate_healing_report(self, output_path="healing_report.html"):
        """Generate an HTML report of healing actions."""
        if not self.healing_history:
            logger.warning("No healing history available for report generation")
            return
            
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Network Self-Healing Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .improvement { color: green; }
            </style>
        </head>
        <body>
            <h1>Network Self-Healing Report</h1>
            <p>Generated on: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
            
            <h2>Healing Actions Summary</h2>
            <table>
                <tr>
                    <th>Anomaly Type</th>
                    <th>Action Taken</th>
                    <th>Before</th>
                    <th>After</th>
                    <th>Improvement</th>
                </tr>
        """
        
        for action in self.healing_history:
            before = action['before']
            after = action['after']
            
            # Calculate improvement percentage (handle different metrics)
            if action['type'] in ['high_packet_loss', 'high_latency']:
                # Lower is better
                improvement = ((before - after) / before) * 100 if before > 0 else 0
                improvement_text = f"+{improvement:.1f}%"
            else:
                # Higher is better
                improvement = ((after - before) / abs(before)) * 100 if before != 0 else 100
                improvement_text = f"+{improvement:.1f}%"
            
            html += f"""
                <tr>
                    <td>{action['type'].replace('_', ' ').title()}</td>
                    <td>{action['action']}</td>
                    <td>{before:.2f}</td>
                    <td>{after:.2f}</td>
                    <td class="improvement">{improvement_text}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html)
            
        logger.info(f"Healing report generated at {output_path}")
        return output_path

# Example usage
if __name__ == "__main__":
    self_healing = SelfHealingMechanism()
    healed_data, healing_actions = self_healing.perform_self_healing()
    
    if healing_actions:
        report_path = self_healing.generate_healing_report()
        print(f"Self-healing complete! Report available at: {report_path}")
    else:
        print("No anomalies detected, network is healthy")