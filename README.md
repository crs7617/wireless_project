# Wireless Network Monitoring and Self-Healing System

A comprehensive system for monitoring wireless network performance, detecting anomalies, and automatically applying self-healing measures to optimize network performance.

## Overview

This project implements a wireless network monitoring and self-healing system that uses AI to detect anomalies in network performance and automatically applies remediation strategies to improve overall network health. The system analyzes key metrics such as signal strength, packet loss, latency, and throughput to identify problematic patterns and takes appropriate action to optimize network performance.


## Features

- **Anomaly Detection**: Uses Isolation Forest algorithm to identify abnormal network behavior
- **Real-time Monitoring**: Tracks key network performance metrics
- **Automated Self-Healing**: Applies remediation strategies such as channel switching, power adjustment, and QoS optimization
- **Visualization**: Generates comprehensive visualizations of network performance and anomalies
- **Performance Reporting**: Creates detailed before/after reports on the effectiveness of healing strategies

## Project Structure

```
wireless_project/
├── backend/
│   ├── ai_model.py         # Anomaly detection model
│   ├── self_heal.py        # Self-healing strategies implementation
│   ├── visualization.py    # Data visualization and reporting functions
│   └── run_wireless_project.py  # Main script to run the project
├── output/
│   ├── reports/            # Performance reports and analysis
│   └── visualizations/     # Generated visualizations and charts
└── wmc.csv                 # Wireless metrics collection data
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/wireless_project.git
cd wireless_project
```

2. Install required dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Ensure you have a `wmc.csv` file in the project root directory with the following columns:
   - timestamp
   - device_id
   - signal_strength
   - packet_loss
   - latency
   - throughput
   - channel
   - transmit_power
   - position_x
   - position_y
   - position_z

## Usage

Run the main script to execute the full pipeline:

```bash
python backend/run_wireless_project.py
```

This will:
1. Load the wireless metrics data
2. Detect anomalies in the network performance
3. Generate visualizations of the anomalies
4. Apply self-healing strategies to optimize performance
5. Generate a report comparing before and after performance

## Self-Healing Process

The self-healing system follows this workflow:

1. **Detection**: Identifies anomalies in network metrics using machine learning
2. **Analysis**: Determines the specific issues based on predefined thresholds
3. **Strategy Selection**: Chooses appropriate healing strategies based on the issues
4. **Implementation**: Simulates applying changes to the network configuration
5. **Verification**: Compares performance before and after applying strategies

### Healing Strategies

The system can apply several healing strategies:

- **Channel Switching**: Moves devices to less congested channels
- **Power Adjustment**: Increases transmit power for devices with weak signals
- **Position Optimization**: Recommends optimal device positioning
- **Bandwidth Allocation**: Optimizes bandwidth distribution
- **QoS Adjustment**: Tunes Quality of Service parameters

## Future Improvements

- Develop a web-based dashboard for network administrators
- Integrate with network management systems for direct configuration changes
- Add support for additional wireless technologies (5G, LoRaWAN, etc.)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

