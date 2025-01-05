import numpy as np
import matplotlib.pyplot as plt

class SimpleEngine:
    def __init__(self, max_power: float, max_rpm: float):
        """
        Initialize a simple engine
        max_power: maximum power in horsepower
        max_rpm: maximum engine RPM
        """
        self.max_power = max_power
        self.max_rpm = max_rpm
        self.idle_rpm = 800
    
    def get_torque(self, rpm: float) -> float:
        """Calculate engine torque at given RPM."""
        if rpm < self.idle_rpm:
            return 0
            
        # Simple torque curve with peak at 60% of max RPM
        peak_rpm = self.max_rpm * 0.6
        torque = self.max_power * 5252 / peak_rpm  # Base torque
        
        # Simple quadratic falloff from peak
        torque *= 1 - 0.8 * ((rpm - peak_rpm) / peak_rpm) ** 2
        
        return max(0, torque)
    
    def get_power(self, rpm: float) -> float:
        """Calculate power at given RPM."""
        return (self.get_torque(rpm) * rpm) / 5252

class SimpleTransmission:
    def __init__(self):
        """Initialize transmission with basic gear ratios."""
        self.gear_ratios = {
            1: 3.5,    # First gear
            2: 2.3,    # Second gear
            3: 1.7,    # Third gear
            4: 1.3,    # Fourth gear
            5: 1.0,    # Fifth gear
            'final': 3.9  # Final drive ratio
        }
    
    def get_speed(self, rpm: float, gear: int, wheel_diameter: float) -> float:
        """
        Calculate vehicle speed
        rpm: engine RPM
        gear: current gear
        wheel_diameter: in meters
        Returns: speed in km/h
        """
        if gear not in self.gear_ratios:
            return 0
            
        total_ratio = self.gear_ratios[gear] * self.gear_ratios['final']
        wheel_rpm = rpm / total_ratio
        
        # Speed calculation
        wheel_circumference = np.pi * wheel_diameter  # meters
        speed_ms = (wheel_rpm * wheel_circumference) / 60
        return speed_ms * 3.6  # Convert to km/h
    
    def get_wheel_torque(self, engine_torque: float, gear: int) -> float:
        """Calculate torque at the wheels."""
        if gear not in self.gear_ratios:
            return 0
            
        total_ratio = self.gear_ratios[gear] * self.gear_ratios['final']
        return engine_torque * total_ratio * 0.85  # 85% efficiency

def plot_performance(engine: SimpleEngine, transmission: SimpleTransmission):
    """Create basic performance plots."""
    rpm_range = np.linspace(engine.idle_rpm, engine.max_rpm, 100)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Engine torque and power
    torques = [engine.get_torque(rpm) for rpm in rpm_range]
    powers = [engine.get_power(rpm) for rpm in rpm_range]
    
    ax1.plot(rpm_range, torques, 'b-', label='Torque (lb-ft)')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(rpm_range, powers, 'r-', label='Power (hp)')
    
    ax1.set_xlabel('Engine Speed (RPM)')
    ax1.set_ylabel('Torque (lb-ft)')
    ax1_twin.set_ylabel('Power (hp)')
    ax1.set_title('Engine Performance')
    ax1.grid(True)
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Plot 2: Speed in each gear
    wheel_diameter = 0.65  # meters (approximately 26 inches)
    for gear in range(1, 6):
        speeds = [transmission.get_speed(rpm, gear, wheel_diameter) 
                 for rpm in rpm_range]
        ax2.plot(rpm_range, speeds, label=f'Gear {gear}')
    
    ax2.set_xlabel('Engine Speed (RPM)')
    ax2.set_ylabel('Vehicle Speed (km/h)')
    ax2.set_title('Speed vs RPM in Different Gears')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('simple_performance.png')
    plt.close()

def main():
    # Create a simple 200hp engine
    engine = SimpleEngine(
        max_power=200,  # 200 horsepower
        max_rpm=6500
    )
    
    # Create transmission
    transmission = SimpleTransmission()
    
    # Print basic info
    print("Engine and Transmission Analysis")
    print("-" * 30)
    print(f"Maximum Power: {engine.max_power} hp")
    print(f"Maximum RPM: {engine.max_rpm}")
    print("\nGear Ratios:")
    for gear, ratio in transmission.gear_ratios.items():
        print(f"Gear {gear}: {ratio}")
    
    # Calculate some example values
    test_rpm = 3000
    test_gear = 3
    wheel_diameter = 0.65
    
    print("\nExample Calculations at 3000 RPM in 3rd gear:")
    print(f"Engine Torque: {engine.get_torque(test_rpm):.1f} lb-ft")
    print(f"Engine Power: {engine.get_power(test_rpm):.1f} hp")
    print(f"Vehicle Speed: {transmission.get_speed(test_rpm, test_gear, wheel_diameter):.1f} km/h")
    
    # Generate performance plots
    plot_performance(engine, transmission)
    print("\nPerformance plots saved as 'simple_performance.png'")

if __name__ == "__main__":
    main()
