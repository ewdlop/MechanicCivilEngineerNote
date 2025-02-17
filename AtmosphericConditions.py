import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
import random
from typing import List, Tuple

@dataclass
class AtmosphericConditions:
    temperature: float  # Celsius
    humidity: float    # 0-1
    pressure: float    # hPa
    wind_speed: float  # m/s
    
    def calculate_instability(self) -> float:
        """Calculate atmospheric instability index"""
        return (self.temperature * self.humidity) / (self.pressure / 1000)

class Thunderstorm:
    def __init__(self, size: float, intensity: float):
        self.size = size  # km
        self.intensity = intensity  # 0-1
        self.lightning_count = 0
        self.rainfall = 0.0  # mm
        self.updraft_speed = 0.0  # m/s
        self.downdraft_speed = 0.0  # m/s
        
    def update(self, conditions: AtmosphericConditions):
        """Update thunderstorm characteristics based on conditions"""
        self.updraft_speed = conditions.wind_speed * (1 + conditions.instability())
        self.downdraft_speed = self.updraft_speed * 0.8
        self.rainfall += self.calculate_precipitation(conditions)
        self.generate_lightning(conditions)
    
    def calculate_precipitation(self, conditions: AtmosphericConditions) -> float:
        """Calculate precipitation rate"""
        return (conditions.humidity * self.intensity * 
                conditions.temperature / 10)
    
    def generate_lightning(self, conditions: AtmosphericConditions):
        """Simulate lightning strikes"""
        if random.random() < self.intensity * conditions.instability():
            self.lightning_count += 1

class Tornado:
    def __init__(self, initial_intensity: float):
        self.intensity = initial_intensity  # Enhanced Fujita scale (0-5)
        self.wind_speed = self.calculate_wind_speed()
        self.path = []
        self.diameter = 0.0  # meters
        self.rotation_speed = 0.0  # m/s
        
    def calculate_wind_speed(self) -> float:
        """Calculate wind speed based on EF scale"""
        base_speed = 64.0  # minimum EF0 wind speed in m/s
        return base_speed * (1 + self.intensity * 0.5)
    
    def update(self, conditions: AtmosphericConditions):
        """Update tornado characteristics"""
        self.wind_speed = self.calculate_wind_speed() * (
            1 + conditions.instability() * 0.2
        )
        self.diameter = 50 + (self.intensity * 100)  # base 50m + intensity factor
        self.rotation_speed = self.wind_speed * 0.7
        
    def calculate_damage_potential(self) -> float:
        """Calculate potential damage on EF scale"""
        return min(5.0, self.intensity * (self.wind_speed / 50))

class WeatherSimulation:
    def __init__(self, grid_size: Tuple[int, int]):
        self.grid_size = grid_size
        self.conditions = self.initialize_conditions()
        self.thunderstorm = None
        self.tornado = None
        self.grid = np.zeros(grid_size)
        
    def initialize_conditions(self) -> AtmosphericConditions:
        """Initialize atmospheric conditions"""
        return AtmosphericConditions(
            temperature=random.uniform(20, 35),
            humidity=random.uniform(0.6, 0.9),
            pressure=random.uniform(980, 1020),
            wind_speed=random.uniform(5, 15)
        )
    
    def spawn_thunderstorm(self):
        """Create a new thunderstorm"""
        self.thunderstorm = Thunderstorm(
            size=random.uniform(5, 15),
            intensity=random.uniform(0.4, 0.9)
        )
    
    def spawn_tornado(self):
        """Create a new tornado if conditions are right"""
        if (self.thunderstorm and 
            self.thunderstorm.updraft_speed > 20 and 
            self.conditions.instability() > 0.7):
            self.tornado = Tornado(random.uniform(0, 5))
    
    def update(self):
        """Update weather conditions and phenomena"""
        # Update base conditions
        self.conditions.temperature += random.uniform(-0.5, 0.5)
        self.conditions.humidity += random.uniform(-0.05, 0.05)
        self.conditions.humidity = max(0, min(1, self.conditions.humidity))
        
        # Update phenomena
        if self.thunderstorm:
            self.thunderstorm.update(self.conditions)
        
        if self.tornado:
            self.tornado.update(self.conditions)
    
    def visualize(self):
        """Create visualization of the weather system"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Thunderstorm visualization
        if self.thunderstorm:
            storm_grid = np.zeros(self.grid_size)
            center = (self.grid_size[0]//2, self.grid_size[1]//2)
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    dist = np.sqrt((i-center[0])**2 + (j-center[1])**2)
                    storm_grid[i,j] = np.exp(-dist/20) * self.thunderstorm.intensity
            
            im1 = ax1.imshow(storm_grid, cmap='Blues')
            ax1.set_title(f'Thunderstorm\nRainfall: {self.thunderstorm.rainfall:.1f}mm\n'
                         f'Lightning Strikes: {self.thunderstorm.lightning_count}')
            plt.colorbar(im1, ax=ax1)
        
        # Tornado visualization
        if self.tornado:
            tornado_grid = np.zeros(self.grid_size)
            center = (self.grid_size[0]//2, self.grid_size[1]//2)
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    dist = np.sqrt((i-center[0])**2 + (j-center[1])**2)
                    tornado_grid[i,j] = np.exp(-dist/5) * self.tornado.intensity
            
            im2 = ax2.imshow(tornado_grid, cmap='RdPu')
            ax2.set_title(f'Tornado\nEF Scale: {self.tornado.intensity:.1f}\n'
                         f'Wind Speed: {self.tornado.wind_speed:.1f} m/s')
            plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        return fig

def run_simulation(frames: int = 100):
    """Run the weather simulation for specified number of frames"""
    sim = WeatherSimulation((50, 50))
    sim.spawn_thunderstorm()
    
    # Only spawn tornado if conditions are right
    if random.random() < 0.3:  # 30% chance
        sim.spawn_tornado()
    
    fig = plt.figure(figsize=(15, 6))
    
    def update(frame):
        plt.clf()
        sim.update()
        sim.visualize()
        plt.title(f'Frame {frame}')
    
    anim = FuncAnimation(fig, update, frames=frames, interval=100)
    return anim

if __name__ == "__main__":
    # Run simulation
    simulation = WeatherSimulation((50, 50))
    simulation.spawn_thunderstorm()
    simulation.spawn_tornado()
    
    # Update and visualize
    for _ in range(10):
        simulation.update()
    simulation.visualize()
    plt.show()
