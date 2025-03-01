import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Generate synthetic normal network traffic (packet_size, latency, jitter)
np.random.seed(42)
num_samples = 1000
packet_size = np.random.normal(loc=500, scale=50, size=num_samples)  # Mean=500B, Std=50B
latency = np.random.normal(loc=50, scale=10, size=num_samples)  # Mean=50ms, Std=10ms
jitter = np.random.normal(loc=5, scale=2, size=num_samples)  # Mean=5ms, Std=2ms

data = np.column_stack((packet_size, latency, jitter))

# Normalize Data
data_min = data.min(axis=0)
data_max = data.max(axis=0)
data = (data - data_min) / (data_max - data_min)

# Define Autoencoder Model
input_dim = data.shape[1]
encoder = keras.Sequential([
    layers.Dense(8, activation="relu", input_shape=(input_dim,)),
    layers.Dense(4, activation="relu"),
])
decoder = keras.Sequential([
    layers.Dense(8, activation="relu", input_shape=(4,)),
    layers.Dense(input_dim, activation="sigmoid"),
])
autoencoder = keras.Sequential([encoder, decoder])

# Compile & Train Model
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(data, data, epochs=20, batch_size=8, shuffle=True)

# Save Model for Future Use
autoencoder.save("anomaly_detector.h5")

# Function to Predict Anomalies
def detect_anomaly(new_data):
    new_data = (new_data - data_min) / (data_max - data_min)  # Normalize
    reconstruction = autoencoder.predict(np.array([new_data]))
    loss = np.mean((new_data - reconstruction) ** 2)
    return loss > 0.05  # Threshold for anomalies

print("✅ Anomaly detection model trained and saved.")
