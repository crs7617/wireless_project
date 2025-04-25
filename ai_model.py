import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

class AnomalyDetector:
    def __init__(self, model_path=None):
        """Initialize the anomaly detector with an optional pre-trained model."""
        self.scaler = StandardScaler()
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.is_trained = True
        else:
            self.model = IsolationForest(contamination=0.05, random_state=42)
            self.is_trained = False
        
    def preprocess_data(self, data):
        """Preprocess wireless network data for anomaly detection."""
        if isinstance(data, pd.DataFrame):
            # Select relevant features for anomaly detection
            features = data[['signal_strength', 'packet_loss', 'latency', 'throughput']]
            return self.scaler.fit_transform(features)
        return data
    
    def train(self, data):
        """Train the anomaly detection model on wireless network data."""
        processed_data = self.preprocess_data(data)
        self.model.fit(processed_data)
        self.is_trained = True
        return self
    
    def detect_anomalies(self, data):
        """Detect anomalies in wireless network data.
        Returns:
            - anomaly_indices: Indices of anomalous data points
            - anomaly_scores: Anomaly scores for each data point (-1 for anomaly, 1 for normal)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() first.")
        
        processed_data = self.preprocess_data(data)
        anomaly_scores = self.model.predict(processed_data)
        anomaly_indices = np.where(anomaly_scores == -1)[0]
        
        # Add detailed anomaly information
        anomaly_details = []
        if isinstance(data, pd.DataFrame):
            for idx in anomaly_indices:
                row = data.iloc[idx]
                anomaly_type = self._classify_anomaly_type(row)
                severity = self._calculate_severity(row)
                anomaly_details.append({
                    'index': idx,
                    'type': anomaly_type,
                    'severity': severity,
                    'data': row.to_dict()
                })
        
        return anomaly_indices, anomaly_scores, anomaly_details
    
    def _classify_anomaly_type(self, data_point):
        """Classify the type of anomaly based on feature values."""
        if data_point['signal_strength'] < -85:
            return "weak_signal"
        elif data_point['packet_loss'] > 10:
            return "high_packet_loss"
        elif data_point['latency'] > 150:
            return "high_latency"
        elif data_point['throughput'] < 5:
            return "low_throughput"
        return "unknown_anomaly"
    
    def _calculate_severity(self, data_point):
        """Calculate the severity of an anomaly from 1 (mild) to 5 (severe)."""
        severity = 1
        
        # Signal strength severity
        if data_point['signal_strength'] < -95:
            severity += 2
        elif data_point['signal_strength'] < -85:
            severity += 1
            
        # Packet loss severity
        if data_point['packet_loss'] > 20:
            severity += 2
        elif data_point['packet_loss'] > 10:
            severity += 1
            
        # Latency severity
        if data_point['latency'] > 200:
            severity += 1
            
        return min(severity, 5)  # Cap at 5
    
    def save_model(self, path="anomaly_model.joblib"):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        joblib.dump(self.model, path)
        return path

# Function to load test data
def load_test_data(file_path):
    """Load test data from CSV file."""
    df = pd.read_csv(file_path)
    return df

# Example usage
if __name__ == "__main__":
    # Load test data
    test_data = load_test_data("wmc.csv")
    
    # Initialize and train anomaly detector
    detector = AnomalyDetector()
    detector.train(test_data)
    
    # Detect anomalies
    anomaly_indices, anomaly_scores, anomaly_details = detector.detect_anomalies(test_data)
    
    # Print results
    print(f"Found {len(anomaly_indices)} anomalies in the data.")
    for detail in anomaly_details[:5]:  # Print first 5 anomalies
        print(f"Anomaly: {detail['type']}, Severity: {detail['severity']}")
    
    # Save the model
    detector.save_model()