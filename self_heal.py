import pandas as pd
import numpy as np
import random
from datetime import datetime

class SelfHealingSystem:
    def __init__(self):
        # Define thresholds for different metrics
        self.thresholds = {
            'signal_strength': -70,  # dBm (higher is better)
            'packet_loss': 5,        # percentage (lower is better)
            'latency': 100,          # ms (lower is better)
            'throughput': 10         # Mbps (higher is better)
        }
        
        # Define healing strategies
        self.strategies = {
            'channel_switch': {
                'description': 'Switch to a less congested channel',
                'impact': {'signal_strength': 5, 'latency': -10, 'packet_loss': -2, 'throughput': 2},
                'applies_to': ['signal_strength', 'latency', 'packet_loss']
            },
            'power_increase': {
                'description': 'Increase transmit power',
                'impact': {'signal_strength': 10, 'throughput': 3},
                'applies_to': ['signal_strength']
            },
            'position_optimization': {
                'description': 'Optimize device positioning',
                'impact': {'signal_strength': 8, 'latency': -5, 'packet_loss': -1, 'throughput': 1},
                'applies_to': ['signal_strength', 'packet_loss']
            },
            'bandwidth_allocation': {
                'description': 'Optimize bandwidth allocation',
                'impact': {'throughput': 5, 'latency': -15},
                'applies_to': ['throughput', 'latency']
            },
            'qos_adjustment': {
                'description': 'Adjust QoS parameters',
                'impact': {'latency': -20, 'packet_loss': -3},
                'applies_to': ['latency', 'packet_loss']
            }
        }

    def identify_issues(self, data):
        """
        Identify issues in the network based on thresholds.
        
        Args:
            data: DataFrame containing network metrics
            
        Returns:
            Dictionary with issues per metric
        """
        issues = {}
        
        # Check which metrics exist in the data
        available_metrics = [m for m in self.thresholds if m in data.columns]
        
        if not available_metrics:
            print("Warning: No recognized metrics found in the data")
            return issues
        
        # Check each metric against its threshold
        for metric in available_metrics:
            if metric in ['signal_strength', 'throughput']:
                # For these metrics, higher is better
                problematic = data[data[metric] < self.thresholds[metric]]
                if len(problematic) > 0:
                    issues[metric] = {
                        'count': len(problematic),
                        'avg_value': problematic[metric].mean(),
                        'threshold': self.thresholds[metric],
                        'description': f"Low {metric}: {len(problematic)} instances below threshold"
                    }
            else:
                # For these metrics, lower is better
                problematic = data[data[metric] > self.thresholds[metric]]
                if len(problematic) > 0:
                    issues[metric] = {
                        'count': len(problematic),
                        'avg_value': problematic[metric].mean(),
                        'threshold': self.thresholds[metric],
                        'description': f"High {metric}: {len(problematic)} instances above threshold"
                    }
        
        return issues

    def generate_healing_plan(self, anomaly_data):
        """
        Generate a healing plan based on the identified issues.
        
        Args:
            anomaly_data: DataFrame containing the anomalous data points
            
        Returns:
            List of healing strategies to apply
        """
        issues = self.identify_issues(anomaly_data)
        
        if not issues:
            return [{'strategy': 'monitoring', 'description': 'No significant issues detected. Continue monitoring.'}]
        
        # Score each strategy based on how well it addresses the issues
        strategy_scores = {name: 0 for name in self.strategies}
        
        for metric, issue in issues.items():
            for strategy_name, strategy in self.strategies.items():
                if metric in strategy['applies_to']:
                    # Higher score for strategies that help with more critical issues
                    strategy_scores[strategy_name] += issue['count']
        
        # Select top strategies
        healing_plan = []
        for strategy_name, score in sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True):
            if score > 0:
                strategy = self.strategies[strategy_name].copy()
                strategy['strategy'] = strategy_name
                healing_plan.append(strategy)
                
                # We'll apply the top 3 strategies at most
                if len(healing_plan) >= 3:
                    break
        
        return healing_plan

    def apply_healing_strategies(self, data, healing_plan):
        """
        Apply healing strategies to the data (simulation).
        
        Args:
            data: Original DataFrame
            healing_plan: List of strategies to apply
            
        Returns:
            Tuple of (optimized_data, list_of_changes)
        """
        optimized_data = data.copy()
        changes_applied = []
        
        # Apply each strategy
        for strategy in healing_plan:
            strategy_name = strategy['strategy']
            description = strategy['description']
            
            if strategy_name == 'monitoring':
                changes_applied.append(description)
                continue
            
            # Each strategy affects different metrics
            impacts = strategy.get('impact', {})
            
            # Apply the impacts to the data
            for metric, impact in impacts.items():
                if metric not in optimized_data.columns:
                    continue
                    
                if metric in ['signal_strength', 'throughput']:
                    # For these metrics, higher is better, so we add the impact
                    optimized_data[metric] = optimized_data[metric] + impact
                else:
                    # For these metrics, lower is better, so we subtract the impact (impact is negative)
                    # But we ensure we don't go below 0
                    optimized_data[metric] = (optimized_data[metric] + impact).clip(0)
            
            # Add randomness to make it more realistic
            for metric in impacts:
                if metric in optimized_data.columns:
                    noise = np.random.normal(0, abs(impacts[metric]) * 0.2, len(optimized_data))
                    optimized_data[metric] = optimized_data[metric] + noise
                    
                    # Ensure values stay in reasonable ranges
                    if metric == 'packet_loss':
                        optimized_data[metric] = optimized_data[metric].clip(0, 100)
                    elif metric == 'signal_strength':
                        optimized_data[metric] = optimized_data[metric].clip(-100, -30)
            
            # If the strategy is channel_switch, actually change the channel values
            if strategy_name == 'channel_switch' and 'channel' in optimized_data.columns:
                # Get current channels
                current_channels = optimized_data['channel'].unique()
                
                # Define potential new channels (1-11 for 2.4GHz, 36-165 for 5GHz)
                all_channels = list(range(1, 12)) + list(range(36, 166, 4))
                
                # Choose channels that are less used currently
                new_channels = [ch for ch in all_channels if ch not in current_channels]
                if not new_channels:  # If all channels are in use, just pick random ones
                    new_channels = all_channels
                
                # Replace problematic channels with new ones
                problematic_devices = set()
                for metric in strategy['applies_to']:
                    if metric in optimized_data.columns:
                        if metric in ['signal_strength', 'throughput']:
                            problematic = optimized_data[optimized_data[metric] < self.thresholds[metric]]
                        else:
                            problematic = optimized_data[optimized_data[metric] > self.thresholds[metric]]
                        problematic_devices.update(problematic['device_id'].unique() if 'device_id' in problematic.columns else [])
                
                # For each problematic device, switch to a better channel
                for device in problematic_devices:
                    if not new_channels:
                        continue
                    new_channel = random.choice(new_channels)
                    device_mask = optimized_data['device_id'] == device
                    optimized_data.loc[device_mask, 'channel'] = new_channel
            
            # If the strategy is power_increase, adjust transmit_power
            if strategy_name == 'power_increase' and 'transmit_power' in optimized_data.columns:
                # Identify devices with signal strength issues
                if 'signal_strength' in optimized_data.columns:
                    problematic = optimized_data[optimized_data['signal_strength'] < self.thresholds['signal_strength']]
                    problematic_devices = problematic['device_id'].unique() if 'device_id' in problematic.columns else []
                    
                    # Increase transmit power for problematic devices
                    for device in problematic_devices:
                        device_mask = optimized_data['device_id'] == device
                        current_power = optimized_data.loc[device_mask, 'transmit_power'].mean()
                        new_power = min(current_power + 3, 20)  # Assuming max power is 20 dBm
                        optimized_data.loc[device_mask, 'transmit_power'] = new_power
            
            # Record the applied change
            changes_applied.append(description)
        
        return optimized_data, changes_applied

    def calculate_overall_improvement(self, before_data, after_data):
        """
        Calculate the overall improvement across all metrics.
        
        Args:
            before_data: Original DataFrame
            after_data: Optimized DataFrame
            
        Returns:
            Dictionary with improvement metrics
        """
        improvement = {}
        
        # Check which metrics exist in both datasets
        metrics = [m for m in self.thresholds if m in before_data.columns and m in after_data.columns]
        
        for metric in metrics:
            before_avg = before_data[metric].mean()
            after_avg = after_data[metric].mean()
            
            if metric in ['signal_strength', 'throughput']:
                # For these metrics, higher is better
                change_pct = ((after_avg - before_avg) / abs(before_avg)) * 100 if before_avg != 0 else 0
                change_direction = "improved" if change_pct > 0 else "degraded"
            else:
                # For these metrics, lower is better
                change_pct = ((before_avg - after_avg) / before_avg) * 100 if before_avg != 0 else 0
                change_direction = "improved" if change_pct > 0 else "degraded"
            
            improvement[metric] = {
                'before': before_avg,
                'after': after_avg,
                'change_pct': change_pct,
                'direction': change_direction
            }
        
        # Calculate overall improvement score (weighted average)
        weights = {
            'signal_strength': 0.25,
            'packet_loss': 0.25,
            'latency': 0.25,
            'throughput': 0.25
        }
        
        total_weight = 0
        weighted_improvement = 0
        
        for metric in improvement:
            if metric in weights:
                total_weight += weights[metric]
                # We take the absolute value since improvement can be in different directions
                weighted_improvement += abs(improvement[metric]['change_pct']) * weights[metric]
        
        if total_weight > 0:
            improvement['overall'] = weighted_improvement / total_weight
        else:
            improvement['overall'] = 0
        
        return improvement