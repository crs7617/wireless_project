# test_self_healing.py

from heal_mode import detect_and_heal

# Sample input data with the required columns used during training
sample = {
    'No.': 1,
    'Time': 0.123456,
    'Length': 60
}

anomaly, action = detect_and_heal(sample)

print("Anomaly Detected:", anomaly)
print("Healing Action:", action)
