import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import time

# Load Pretrained Model
autoencoder = load_model("anomaly_detector.h5")

# Generate Random Network Traffic for Monitoring
def generate_traffic():
    packet_size = np.random.normal(loc=500, scale=50)  # Mean=500B, Std=50B
    latency = np.random.normal(loc=50, scale=10)  # Mean=50ms, Std=10ms
    jitter = np.random.normal(loc=5, scale=2)  # Mean=5ms, Std=2ms
    return np.array([packet_size, latency, jitter], dtype=np.float32)

# Self-Healing Function (Block IP, Restart Service, Alert Admin)
def self_heal():
    print("🚨 Anomaly detected! Taking self-healing action...")

    # Example actions:
    os.system("netsh advfirewall firewall add rule name='Block Suspicious IP' dir=in action=block remoteip=192.168.1.100")
    os.system("systemctl restart network-manager")  # Restart network service (Linux)

    # Log Anomaly
    with open("anomaly_log.txt", "a") as log_file:
        log_file.write("Anomaly detected & self-healed.\n")

# Monitor Traffic and Detect Anomalies
while True:
    latest_data = generate_traffic().reshape(1, -1)
    reconstruction = autoencoder.predict(latest_data)
    loss = np.mean((latest_data - reconstruction) ** 2)

    if loss > 0.05:  # If anomaly is detected
        self_heal()
    
    time.sleep(2)  # Simulate real-time monitoring
