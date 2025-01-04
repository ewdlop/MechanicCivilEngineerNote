"""
===================================================================================
            Nano-Enhanced Civil Engineering Project Library
===================================================================================
Created by: Claude (Anthropic AI Assistant)
Model: Claude 3.5 Sonnet
Version: 1.0
Created: 2024

A comprehensive library combining nanomechanics with civil engineering applications.
Focus areas: Smart materials, self-healing concrete, structural health monitoring.
===================================================================================
"""

import numpy as np
from scipy import integrate
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

class SmartMaterialSimulator:
    """Simulates smart materials with nano-enhanced properties."""
    
    def __init__(self):
        self.nano_concentration = None  # kg/m³
        self.base_strength = None      # MPa
        self.self_healing_capacity = None  # healing events per year
        
    def set_material_properties(self, 
                              nano_conc: float, 
                              base_strength: float, 
                              healing_cap: float):
        """Set basic material properties."""
        self.nano_concentration = nano_conc
        self.base_strength = base_strength
        self.self_healing_capacity = healing_cap
        
    def calculate_enhanced_strength(self, age: float) -> float:
        """Calculate strength enhancement from nano-additives."""
        enhancement_factor = 1 + (0.15 * self.nano_concentration / 100)
        aging_factor = 1 - (0.02 * np.log(age + 1))
        return self.base_strength * enhancement_factor * aging_factor
    
    def predict_crack_healing(self, 
                            crack_width: float, 
                            time: float) -> float:
        """Predict crack healing progress over time."""
        healing_rate = self.self_healing_capacity * np.exp(-time/365)
        healed_width = crack_width * (1 - np.exp(-healing_rate * time))
        return healed_width

class SelfHealingConcrete:
    """Manages self-healing concrete properties and behavior."""
    
    def __init__(self):
        self.capsule_density = None    # capsules/m³
        self.healing_agent_volume = None  # mL/capsule
        self.activation_threshold = None  # strain threshold
        
    def set_healing_properties(self, 
                             density: float, 
                             volume: float, 
                             threshold: float):
        """Set self-healing properties."""
        self.capsule_density = density
        self.healing_agent_volume = volume
        self.activation_threshold = threshold
        
    def calculate_healing_capacity(self, 
                                 damage_volume: float) -> float:
        """Calculate total healing capacity for given damage."""
        available_agent = self.capsule_density * self.healing_agent_volume
        efficiency = 0.75  # typical efficiency factor
        return min(1.0, (available_agent * efficiency) / damage_volume)
    
    def predict_strength_recovery(self, 
                                initial_strength: float, 
                                damage_level: float, 
                                time: float) -> float:
        """Predict strength recovery after damage."""
        healing_factor = 1 - np.exp(-time/30)  # 30 days characteristic time
        recovery = damage_level * healing_factor
        return initial_strength * (1 - damage_level + recovery)

class StructuralHealthMonitor:
    """Monitors structural health using nano-sensors."""
    
    def __init__(self):
        self.sensor_network = {}  # Dictionary of sensor locations and readings
        self.baseline_readings = {}
        self.alarm_thresholds = {}
        
    def add_sensor(self, 
                  location: Tuple[float, float, float], 
                  sensor_type: str):
        """Add a new sensor to the network."""
        self.sensor_network[location] = {
            'type': sensor_type,
            'readings': [],
            'status': 'active'
        }
        
    def record_reading(self, 
                      location: Tuple[float, float, float], 
                      value: float, 
                      timestamp: float):
        """Record a sensor reading."""
        if location in self.sensor_network:
            self.sensor_network[location]['readings'].append({
                'value': value,
                'time': timestamp
            })
            
    def analyze_strain_distribution(self, 
                                  time_point: float) -> Dict:
        """Analyze strain distribution across structure."""
        strain_map = {}
        for loc, sensor in self.sensor_network.items():
            readings = [r for r in sensor['readings'] if r['time'] <= time_point]
            if readings:
                strain_map[loc] = readings[-1]['value']
        return strain_map
    
    def detect_anomalies(self, 
                        threshold: float = 0.1) -> List[Tuple]:
        """Detect structural anomalies from sensor data."""
        anomalies = []
        for loc, sensor in self.sensor_network.items():
            if sensor['readings']:
                baseline = np.mean(sensor['readings'][:10]['value'])
                current = sensor['readings'][-1]['value']
                if abs(current - baseline) > threshold:
                    anomalies.append((loc, current - baseline))
        return anomalies

class NanoReinforcedStructure:
    """Manages nano-reinforced structural elements."""
    
    def __init__(self):
        self.reinforcement_type = None
        self.reinforcement_ratio = None
        self.base_material = None
        
    def set_reinforcement(self, 
                         r_type: str, 
                         ratio: float, 
                         material: str):
        """Set reinforcement properties."""
        self.reinforcement_type = r_type
        self.reinforcement_ratio = ratio
        self.base_material = material
        
    def calculate_enhanced_properties(self) -> Dict:
        """Calculate enhanced material properties."""
        enhancement_factors = {
            'CNT': {'strength': 1.4, 'stiffness': 1.3, 'durability': 1.5},
            'Graphene': {'strength': 1.5, 'stiffness': 1.4, 'durability': 1.6},
            'Nano-silica': {'strength': 1.2, 'stiffness': 1.1, 'durability': 1.3}
        }
        
        base_properties = {
            'concrete': {'strength': 30, 'stiffness': 30e9, 'durability': 50},
            'steel': {'strength': 400, 'stiffness': 200e9, 'durability': 75}
        }
        
        factors = enhancement_factors.get(self.reinforcement_type, 
                                       {'strength': 1, 'stiffness': 1, 'durability': 1})
        base = base_properties.get(self.base_material, 
                                 {'strength': 1, 'stiffness': 1, 'durability': 1})
        
        return {
            'strength': base['strength'] * factors['strength'] * (1 + self.reinforcement_ratio),
            'stiffness': base['stiffness'] * factors['stiffness'] * (1 + self.reinforcement_ratio),
            'durability': base['durability'] * factors['durability'] * (1 + self.reinforcement_ratio)
        }

def main():
    """Example implementation of nano-enhanced civil engineering project."""
    
    # Initialize smart material simulation
    smart_material = SmartMaterialSimulator()
    smart_material.set_material_properties(
        nano_conc=5,  # 5 kg/m³ nano-additives
        base_strength=40,  # 40 MPa base strength
        healing_cap=10  # 10 healing events per year
    )
    
    # Initialize self-healing concrete
    concrete = SelfHealingConcrete()
    concrete.set_healing_properties(
        density=1000,  # 1000 capsules/m³
        volume=0.5,    # 0.5 mL per capsule
        threshold=0.0001  # activation at 0.01% strain
    )
    
    # Setup structural health monitoring
    monitor = StructuralHealthMonitor()
    # Add sensors at key locations
    monitor.add_sensor((0, 0, 0), "strain")
    monitor.add_sensor((5, 0, 0), "strain")
    monitor.add_sensor((10, 0, 0), "strain")
    
    # Initialize nano-reinforced structure
    structure = NanoReinforcedStructure()
    structure.set_reinforcement(
        r_type="CNT",
        ratio=0.02,  # 2% reinforcement ratio
        material="concrete"
    )
    
    # Example calculations
    enhanced_strength = smart_material.calculate_enhanced_strength(age=180)
    print(f"Enhanced strength after 180 days: {enhanced_strength:.2f} MPa")
    
    healing_capacity = concrete.calculate_healing_capacity(damage_volume=0.1)
    print(f"Healing capacity: {healing_capacity*100:.1f}%")
    
    enhanced_properties = structure.calculate_enhanced_properties()
    print("Enhanced material properties:")
    for prop, value in enhanced_properties.items():
        print(f"{prop}: {value:.2f}")

if __name__ == "__main__":
    main()
