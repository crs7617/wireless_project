import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class AnomalyDetector:
    def __init__(self, contamination=0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def preprocess_data(self, data):
        # Get the column names from the actual dataframe
        columns = data.columns
        
        # Define the feature sets we want to use for anomaly detection
        metric_columns = []
        for col in ['signal_strength', 'packet_loss', 'latency', 'throughput']:
            if col in columns:
                metric_columns.append(col)
        
        # If the columns weren't found, try alternative names or use available numeric columns
        if not metric_columns:
            # Look for columns containing similar keywords
            for col in columns:
                if any(keyword in col.lower() for keyword in ['signal', 'strength', 'packet', 'loss', 'latency', 'through']):
                    metric_columns.append(col)
            
            # If still no columns found, use numeric columns (excluding timestamps and IDs)
            if not metric_columns:
                for col in columns:
                    if data[col].dtype in [np.int64, np.float64] and 'id' not in col.lower() and 'time' not in col.lower():
                        metric_columns.append(col)
        
        print(f"Using these columns for anomaly detection: {metric_columns}")
        
        features = data[metric_columns]
        return features

    def train(self, data):
        processed_data = self.preprocess_data(data)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(processed_data)
        
        # Train the model
        self.model.fit(scaled_data)
        self.is_trained = True
        
        return self

    def detect_anomalies(self, data):
        if not self.is_trained:
            raise Exception("Model is not trained yet. Call train() first.")
        
        processed_data = self.preprocess_data(data)
        scaled_data = self.scaler.transform(processed_data)
        
        # Predict anomalies
        # -1 for anomalies, 1 for normal data
        predictions = self.model.predict(scaled_data)
        
        # Convert to boolean (True for anomalies)
        anomalies = predictions == -1
        
        # Calculate anomaly scores
        scores = self.model.score_samples(scaled_data)
        # Lower scores indicate more anomalous behavior
        
        return anomalies, scores

    def get_anomaly_details(self, data, anomalies, scores):
        anomaly_indices = np.where(anomalies)[0]
        
        results = []
        for idx in anomaly_indices:
            row = data.iloc[idx]
            score = scores[idx]
            
            # Create a result dictionary with all relevant information
            result = {
                'index': idx,
                'score': score,
                'data': row.to_dict()
            }
            
            results.append(result)
            
        return results