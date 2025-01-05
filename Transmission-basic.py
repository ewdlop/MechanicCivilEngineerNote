import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from enum import Enum

class TransmissionType(Enum):
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    CVT = "cvt"
    DCT = "dual_clutch"

class Engine:
    def __init__(self, displacement: float, max_rpm: float, max_power: float):
        """
        Initialize engine parameters
        displacement: in liters
        max_rpm: maximum engine RPM
        max_power: maximum power in horsepower
        """
        self.displacement = displacement
        self.max_rpm = max_rpm
        self.max_power = max_power
        self.current_rpm = 1000
        self.idle_rpm = 800
        
    def calculate_torque(self, rpm: float) -> float:
        """Calculate engine torque using a simplified torque curve."""
        # Simplified torque curve modeling
        peak_torque_rpm = self.max_rpm * 0.6
        
        if rpm < self.idle_rpm:
            return 0
        
        # Quadratic torque curve with peak at 60% of max RPM
        torque = (-1.5 * (rpm - peak_torque_rpm)**2 / peak_torque_rpm**2 + 1) * \
                 (self.max_power * 5252 / peak_torque_rpm)
        
        return max(0, torque)
    
    def get_power(self, rpm: float) -> float:
        """Calculate engine power at given RPM."""
        return self.calculate_torque(rpm) * rpm / 5252

class Transmission:
    def __init__(self, transmission_type: TransmissionType):
        self.type = transmission_type
        self.current_gear = 1
        
        # Gear ratios (typical values)
        if transmission_type == TransmissionType.MANUAL:
            self.gear_ratios = {
                1: 3.545,
                2: 2.285,
                3: 1.678,
                4: 1.159,
                5: 0.851,
                6: 0.672,
                'reverse': -3.168,
                'final_drive': 4.100
            }
        elif transmission_type == TransmissionType.AUTOMATIC:
            self.gear_ratios = {
                1: 3.520,
                2: 2.290,
                3: 1.520,
                4: 1.000,
                5: 0.752,
                6: 0.625,
                'reverse': -2.940,
                'final_drive': 3.750
            }
        
        # CVT uses ratio range instead of fixed gears
        self.cvt_ratio_range = (0.5, 2.5)
    
    def get_gear_ratio(self, gear: int) -> float:
        """Get total gear ratio including final drive."""
        if self.type == TransmissionType.CVT:
            # For CVT, calculate continuous ratio based on speed/load
            ratio = np.interp(gear, [1, 6], self.cvt_ratio_range)
            return ratio * 4.0  # Typical CVT final drive ratio
        
        return self.gear_ratios[gear] * self.gear_ratios['final_drive']
    
    def calculate_wheel_torque(self, engine_torque: float, gear: int) -> float:
        """Calculate torque at the wheels."""
        ratio = self.get_gear_ratio(gear)
        return engine_torque * ratio * 0.85  # Assuming 85% drivetrain efficiency

class Car:
    def __init__(self, engine: Engine, transmission: Transmission):
        self.engine = engine
        self.transmission = transmission
        self.wheel_diameter = 0.65  # meters
        self.mass = 1500  # kg
        self.drag_coefficient = 0.30
        self.frontal_area = 2.2  # m²
        
    def calculate_speed(self, engine_rpm: float, gear: int) -> float:
        """Calculate vehicle speed in km/h given engine RPM and gear."""
        if engine_rpm < self.engine.idle_rpm:
            return 0
            
        wheel_rpm = engine_rpm / self.transmission.get_gear_ratio(gear)
        wheel_circumference = np.pi * self.wheel_diameter
        speed_ms = wheel_rpm * wheel_circumference / 60
        return speed_ms * 3.6  # Convert to km/h
    
    def calculate_acceleration(self, speed: float, wheel_torque: float) -> float:
        """Calculate vehicle acceleration given current speed and wheel torque."""
        # Air resistance
        air_resistance = 0.5 * 1.225 * self.drag_coefficient * \
                        self.frontal_area * (speed/3.6)**2
        
        # Rolling resistance (simplified)
        rolling_resistance = self.mass * 9.81 * 0.015
        
        # Net force
        driving_force = wheel_torque / (self.wheel_diameter/2)
        net_force = driving_force - air_resistance - rolling_resistance
        
        # Acceleration (F = ma)
        return net_force / self.mass

def plot_performance_curves(car: Car):
    """Generate performance plots for the car."""
    rpm_range = np.linspace(car.engine.idle_rpm, car.engine.max_rpm, 100)
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Torque and Power curves
    torques = [car.engine.calculate_torque(rpm) for rpm in rpm_range]
    powers = [car.engine.get_power(rpm) for rpm in rpm_range]
    
    ax1.plot(rpm_range, torques, 'b-', label='Torque (lb-ft)')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(rpm_range, powers, 'r-', label='Power (hp)')
    
    ax1.set_xlabel('Engine Speed (RPM)')
    ax1.set_ylabel('Torque (lb-ft)')
    ax1_twin.set_ylabel('Power (hp)')
    ax1.set_title('Engine Performance Curves')
    ax1.grid(True)
    
    # Speed in each gear
    speeds = {}
    for gear in range(1, 7):
        speeds[gear] = [car.calculate_speed(rpm, gear) for rpm in rpm_range]
        ax2.plot(rpm_range, speeds[gear], label=f'Gear {gear}')
    
    ax2.set_xlabel('Engine Speed (RPM)')
    ax2.set_ylabel('Vehicle Speed (km/h)')
    ax2.set_title('Speed vs RPM in Different Gears')
    ax2.grid(True)
    ax2.legend()
    
    # Acceleration capability
    for gear in range(1, 7):
        wheel_torques = [car.transmission.calculate_wheel_torque(
            car.engine.calculate_torque(rpm), gear) for rpm in rpm_range]
        accelerations = [car.calculate_acceleration(speed, torque) 
                        for speed, torque in zip(speeds[gear], wheel_torques)]
        ax3.plot(speeds[gear], accelerations, label=f'Gear {gear}')
    
    ax3.set_xlabel('Vehicle Speed (km/h)')
    ax3.set_ylabel('Acceleration (m/s²)')
    ax3.set_title('Acceleration Capability')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('car_performance.png')
    plt.close()

def main():
    # Create example car with a 3.0L engine
    engine = Engine(
        displacement=3.0,
        max_rpm=7500,
        max_power=300  # 300 hp
    )
    
    # Test different transmission types
    for trans_type in TransmissionType:
        print(f"\nAnalyzing {trans_type.value} transmission...")
        
        transmission = Transmission(trans_type)
        car = Car(engine, transmission)
        
        # Calculate and display performance metrics
        print("\nPerformance Analysis:")
        print(f"Maximum torque: {max(engine.calculate_torque(rpm) for rpm in range(1000, 7501)): .1f} lb-ft")
        
        # Calculate top speed in each gear
        for gear in range(1, 7):
            top_speed = car.calculate_speed(engine.max_rpm, gear)
            print(f"Top speed in gear {gear}: {top_speed:.1f} km/h")
        
        # Generate performance plots
        plot_performance_curves(car)
        print(f"\nPerformance curves saved as 'car_performance.png'")
        
        # Calculate 0-100 km/h time (simplified)
        speed = 0
        time = 0
        gear = 1
        dt = 0.1  # seconds
        
        while speed < 100 and time < 30:  # Maximum 30 seconds
            rpm = speed * 60 * transmission.get_gear_ratio(gear) / \
                  (np.pi * car.wheel_diameter * 3.6)
            
            if rpm > engine.max_rpm * 0.95 and gear < 6:
                gear += 1
                
            torque = engine.calculate_torque(rpm)
            wheel_torque = transmission.calculate_wheel_torque(torque, gear)
            acceleration = car.calculate_acceleration(speed, wheel_torque)
            
            speed += acceleration * dt * 3.6  # Convert to km/h
            time += dt
        
        print(f"0-100 km/h time: {time:.1f} seconds")

if __name__ == "__main__":
    main()
