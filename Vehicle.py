import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from enum import Enum

class FuelType(Enum):
    GASOLINE = "gasoline"
    DIESEL = "diesel"
    HYBRID = "hybrid"

class DrivetrainType(Enum):
    FWD = "front_wheel_drive"
    RWD = "rear_wheel_drive"
    AWD = "all_wheel_drive"

class CombustionEngine:
    def __init__(self, displacement: float, cylinders: int, fuel_type: FuelType,
                 compression_ratio: float = 10.5):
        self.displacement = displacement  # Liters
        self.cylinders = cylinders
        self.fuel_type = fuel_type
        self.compression_ratio = compression_ratio
        
        # Engine characteristics
        self.bore = np.power((displacement * 1000) / (cylinders * np.pi * 0.89), 1/3)  # mm
        self.stroke = self.bore * 0.89  # Typical bore/stroke ratio
        self.max_rpm = 7500 if fuel_type == FuelType.GASOLINE else 5500
        self.idle_rpm = 800 if fuel_type == FuelType.GASOLINE else 700
        
        # Thermal efficiency calculation
        self.thermal_efficiency = 1 - (1 / np.power(compression_ratio, 0.4))
        
        # Power and torque characteristics
        self.max_bmep = 1000 if fuel_type == FuelType.GASOLINE else 1200  # kPa
        self.volumetric_efficiency = {}  # RPM: efficiency mapping
        self._calculate_volumetric_efficiency()
    
    def _calculate_volumetric_efficiency(self):
        """Calculate volumetric efficiency across RPM range."""
        rpms = np.linspace(self.idle_rpm, self.max_rpm, 50)
        peak_ve_rpm = self.max_rpm * 0.65
        
        for rpm in rpms:
            # Bell curve for volumetric efficiency
            ve = 0.85 * np.exp(-((rpm - peak_ve_rpm) / 2000)**2)
            self.volumetric_efficiency[rpm] = ve + 0.1
    
    def calculate_power(self, rpm: float) -> float:
        """Calculate engine power in horsepower."""
        if rpm < self.idle_rpm or rpm > self.max_rpm:
            return 0
            
        ve = self.volumetric_efficiency[min(self.volumetric_efficiency.keys(), 
                                          key=lambda x: abs(x-rpm))]
        
        # Power calculation based on displacement, BMEP, and efficiency
        power_kw = (self.displacement * self.max_bmep * ve * rpm) / (120 * 1000)
        return power_kw * 1.341  # Convert to HP
    
    def calculate_torque(self, rpm: float) -> float:
        """Calculate engine torque in lb-ft."""
        power_hp = self.calculate_power(rpm)
        if rpm > 0:
            return (power_hp * 5252) / rpm
        return 0

class HybridSystem:
    def __init__(self, electric_power: float, battery_capacity: float):
        self.electric_power = electric_power  # kW
        self.battery_capacity = battery_capacity  # kWh
        self.battery_charge = battery_capacity
        self.max_regen = electric_power * 0.3  # 30% regen capability
        
    def get_electric_assist(self, speed: float, throttle: float) -> float:
        """Calculate electric motor assist power."""
        if self.battery_charge <= 0:
            return 0
        
        # Electric assist varies with speed and throttle
        assist = self.electric_power * throttle * (1 - speed/200)
        self.battery_charge -= assist * 0.001  # Rough energy consumption
        return max(0, assist * 1.341)  # Convert to HP
    
    def calculate_regen(self, braking_force: float) -> float:
        """Calculate regenerative braking power."""
        regen = min(self.max_regen, braking_force * 0.5)
        self.battery_charge = min(self.battery_capacity,
                                self.battery_charge + regen * 0.001)
        return regen

class AdvancedTransmission:
    def __init__(self, gear_ratios: Dict[int, float], efficiency: float = 0.92):
        self.gear_ratios = gear_ratios
        self.efficiency = efficiency
        self.current_gear = 1
        self.shift_points = {}
        self._calculate_shift_points()
    
    def _calculate_shift_points(self):
        """Calculate optimal shift points based on gear ratios."""
        for gear in range(1, max(self.gear_ratios.keys())):
            ratio_change = self.gear_ratios[gear] / self.gear_ratios[gear + 1]
            # Shift up at 80% of redline for optimal acceleration
            self.shift_points[gear] = {'up': 0.8, 'down': 0.4}
    
    def get_optimal_gear(self, speed: float, rpm: float, throttle: float) -> int:
        """Determine optimal gear based on current conditions."""
        current_ratio = self.gear_ratios[self.current_gear]
        
        # Check upshift
        if self.current_gear < max(self.gear_ratios.keys()):
            if rpm > self.shift_points[self.current_gear]['up'] * 7500:
                return self.current_gear + 1
                
        # Check downshift
        if self.current_gear > 1:
            if rpm < self.shift_points[self.current_gear-1]['down'] * 7500:
                return self.current_gear - 1
        
        return self.current_gear

class Tire:
    def __init__(self, width: int, aspect_ratio: int, diameter: int):
        """
        width: tire width in mm
        aspect_ratio: tire aspect ratio (%)
        diameter: wheel diameter in inches
        """
        self.width = width
        self.aspect_ratio = aspect_ratio
        self.diameter = diameter
        self.pressure = 32  # PSI
        self.temperature = 20  # Celsius
        
        # Calculate tire characteristics
        self.radius = (diameter * 25.4 + 2 * width * aspect_ratio / 100) / 2000  # meters
        self.contact_patch = width * width * aspect_ratio / (1000 * pressure)  # m²
        
    def calculate_friction(self, speed: float, load: float) -> float:
        """Calculate tire friction coefficient."""
        # Simplified Pacejka formula
        slip_angle = min(12, speed * 0.01)  # Simplified slip angle calculation
        Fz = load * 9.81  # Normal force in N
        
        # Temperature effect on friction
        temp_factor = 1 + (self.temperature - 20) * 0.002
        
        # Basic friction coefficient
        mu = 1.0 * temp_factor * (1 - np.exp(-3.0 * slip_angle/12))
        
        # Load sensitivity
        mu *= np.power(Fz/4000, -0.1)
        
        return mu

class EnhancedVehicle:
    def __init__(self, engine: CombustionEngine, transmission: AdvancedTransmission,
                 drivetrain: DrivetrainType, hybrid: HybridSystem = None):
        self.engine = engine
        self.transmission = transmission
        self.drivetrain = drivetrain
        self.hybrid = hybrid
        
        # Vehicle characteristics
        self.mass = 1500  # kg
        self.drag_cd = 0.30
        self.frontal_area = 2.2  # m²
        
        # Drivetrain losses
        self.drivetrain_losses = {
            DrivetrainType.FWD: 0.85,
            DrivetrainType.RWD: 0.87,
            DrivetrainType.AWD: 0.80
        }
        
        # Tires (example: 225/45R17)
        self.tires = Tire(225, 45, 17)
    
    def calculate_total_power(self, speed: float, throttle: float) -> Dict[str, float]:
        """Calculate total power output including hybrid if equipped."""
        rpm = speed * 60 * self.transmission.gear_ratios[self.transmission.current_gear] / \
              (2 * np.pi * self.tires.radius)
              
        ice_power = self.engine.calculate_power(rpm) * throttle
        
        total_power = {
            'ice': ice_power,
            'electric': 0,
            'total': ice_power
        }
        
        if self.hybrid:
            electric_power = self.hybrid.get_electric_assist(speed, throttle)
            total_power['electric'] = electric_power
            total_power['total'] += electric_power
        
        # Apply drivetrain losses
        total_power['wheel'] = total_power['total'] * \
                              self.drivetrain_losses[self.drivetrain]
        
        return total_power

def plot_enhanced_performance(vehicle: EnhancedVehicle):
    """Generate comprehensive performance plots."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # RPM range for plots
    rpm_range = np.linspace(vehicle.engine.idle_rpm, vehicle.engine.max_rpm, 100)
    
    # 1. Engine Power and Torque curves
    power_curve = [vehicle.engine.calculate_power(rpm) for rpm in rpm_range]
    torque_curve = [vehicle.engine.calculate_torque(rpm) for rpm in rpm_range]
    
    ax1.plot(rpm_range, torque_curve, 'b-', label='Torque (lb-ft)')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(rpm_range, power_curve, 'r-', label='Power (HP)')
    ax1.set_title('Engine Output')
    ax1.set_xlabel('RPM')
    ax1.set_ylabel('Torque (lb-ft)')
    ax1_twin.set_ylabel('Power (HP)')
    ax1.grid(True)
    
    # 2. Speed vs RPM in each gear
    speeds = np.linspace(0, 250, 100)
    for gear in vehicle.transmission.gear_ratios:
        if gear != 'final_drive' and gear != 'reverse':
            gear_speeds = [speed for speed in speeds]
            ax2.plot(rpm_range, gear_speeds, label=f'Gear {gear}')
    
    ax2.set_title('Speed vs RPM per Gear')
    ax2.set_xlabel('RPM')
    ax2.set_ylabel('Speed (km/h)')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Traction curves
    for gear in range(1, 7):
        wheel_forces = []
        for speed in speeds:
            power = vehicle.calculate_total_power(speed, 1.0)
            force = power['wheel'] * 746 / max(1, speed * 0.277778)  # Convert HP to Watts
            wheel_forces.append(force)
        ax3.plot(speeds, wheel_forces, label=f'Gear {gear}')
    
    ax3.set_title('Traction Force vs Speed')
    ax3.set_xlabel('Speed (km/h)')
    ax3.set_ylabel('Force (N)')
    ax3.grid(True)
    ax3.legend()
    
    # 4. Hybrid power distribution (if equipped)
    if vehicle.hybrid:
        ice_power = []
        electric_power = []
        for speed in speeds:
            power = vehicle.calculate_total_power(speed, 0.7)  # 70% throttle
            ice_power.append(power['ice'])
            electric_power.append(power['electric'])
        
        ax4.stackplot(speeds, [ice_power, electric_power],
                     labels=['ICE', 'Electric'])
        ax4.set_title('Power Distribution')
        ax4.set_xlabel('Speed (km/h)')
        ax4.set_ylabel('Power (HP)')
        ax4.grid(True)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No Hybrid System', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('enhanced_vehicle_performance.png')
    plt.close()

def main():
    # Create example vehicle with hybrid system
    engine = CombustionEngine(
        displacement=2.5,
        cylinders=4,
        fuel_type=FuelType.HYBRID,
        compression_ratio=13.0
    )
    
    transmission = AdvancedTransmission({
        1: 3.545,
        2: 2.285,
        3: 1.678,
        4: 1.159,
        5: 0.851,
        6: 0.672,
        'reverse': -3.168,
        'final_drive': 4.100
    })
    
    hybrid = HybridSystem(
        electric_power=60,  # 60 kW electric motor
        battery_capacity=8.8  # 8.8 kWh battery
    )
    
    vehicle = EnhancedVehicle(
        engine=engine,
        transmission=transmission,
        drivetrain=DrivetrainType.AWD,
        hybrid=hybrid
    )
    
    # Generate performance analysis
    print("Vehicle Analysis:")
    print(f"Engine: {engine.displacement}L {engine.cylinders}-cylinder {engine.fuel_type.value}")
    print(f"Max ICE Power: {max(engine.calculate_power(rpm) for rpm in range(1000, 7501)):.1f} HP")
    print(f"Electric Power: {hybrid.electric_power * 1.341:.1f} HP")
    print(f"Drivetrain Efficiency: {vehicle.drivetrain_losses[vehicle.drivetrain]*100:.1f}%")
    
    # Generate plots
    plot_enhanced_performance(vehicle)
    print("\nPerformance plots saved as 'enhanced_vehicle_performance.png'")

if __name__ == "__main__":
    main()
