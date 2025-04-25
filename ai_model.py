import tensorflow as tf

MODEL_PATH = "anomaly_detector.h5"

def load_model():
    """Loads the pre-trained anomaly detection model."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
