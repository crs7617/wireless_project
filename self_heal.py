from ai_model import load_model
import numpy as np

model = load_model()

def detect_anomalies(data):
    """Detects anomalies in network traffic."""
    if model:
        predictions = model.predict(np.array(data).reshape(1, -1))
        if predictions[0] > 0.8:  # Assuming threshold for anomaly
            print("⚠️ Anomaly detected! Attempting self-healing...")
            self_heal()
        else:
            print("✅ Normal network behavior.")

def self_heal():
    """Attempts network recovery."""
    print("Restarting affected services...")
