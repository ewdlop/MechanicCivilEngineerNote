import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import random

@dataclass
class Tetrapod:
    x: float  # x-coordinate
    y: float  # y-coordinate
    size: float  # size/mass in tons
    orientation: float  # rotation angle in degrees
    
    def calculate_stability(self, wave_height: float) -> float:
        """Calculate stability factor based on size and wave conditions"""
        return (self.size * 9.81) / (1000 * wave_height ** 2)

class TetrapodBarrier:
    def __init__(self, length: float, rows: int):
        self.length = length
        self.rows = rows
        self.tetrapods: List[Tetrapod] = []
        self._place_tetrapods()
    
    def _place_tetrapods(self):
        """Place tetrapods in an interlocking pattern"""
        for row in range(self.rows):
            offset = 0 if row % 2 == 0 else 2.5
            for x in np.arange(offset, self.length, 5):
                size = random.uniform(2, 10)  # 2-10 ton range
                orientation = random.uniform(0, 360)
                self.tetrapods.append(Tetrapod(x, row * 4, size, orientation))
    
    def evaluate_protection(self, wave_height: float) -> float:
        """Evaluate overall protection level of the barrier"""
        return np.mean([t.calculate_stability(wave_height) for t in self.tetrapods])

@dataclass
class LeveeSection:
    height: float
    width: float
    material: str
    permeability: float
    
    def calculate_safety_factor(self, water_level: float) -> float:
        """Calculate safety factor based on water level and levee properties"""
        if water_level > self.height:
            return 0.0
        base_factor = (self.width / self.height) * (1 - self.permeability)
        water_pressure = 1 - (water_level / self.height)
        return base_factor * water_pressure

class LeveeSystem:
    def __init__(self, length: float, sections: int):
        self.length = length
        self.sections = sections
        self.levee_sections: List[LeveeSection] = []
        self._construct_levee()
    
    def _construct_levee(self):
        """Initialize levee sections with varying properties"""
        section_length = self.length / self.sections
        for _ in range(self.sections):
            height = random.uniform(3, 6)  # 3-6 meters
            width = height * random.uniform(2, 4)  # width-to-height ratio
            permeability = random.uniform(0.1, 0.3)
            self.levee_sections.append(
                LeveeSection(height, width, "earth", permeability)
            )
    
    def evaluate_flood_protection(self, water_level: float) -> Tuple[float, List[float]]:
        """Evaluate flood protection capability of the levee system"""
        safety_factors = [
            section.calculate_safety_factor(water_level)
            for section in self.levee_sections
        ]
        return min(safety_factors), safety_factors

def visualize_structures(tetrapod_barrier: TetrapodBarrier, levee: LeveeSystem):
    """Visualize both protective structures"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot tetrapod barrier
    for tetrapod in tetrapod_barrier.tetrapods:
        ax1.scatter(tetrapod.x, tetrapod.y, s=tetrapod.size*20, alpha=0.6)
    ax1.set_title('Tetrapod Barrier Layout')
    ax1.set_xlabel('Length (m)')
    ax1.set_ylabel('Width (m)')
    
    # Plot levee system
    x = np.linspace(0, levee.length, len(levee.levee_sections))
    heights = [section.height for section in levee.levee_sections]
    ax2.plot(x, heights, 'b-', label='Levee Height')
    ax2.fill_between(x, heights, alpha=0.3)
    ax2.set_title('Levee System Profile')
    ax2.set_xlabel('Length (m)')
    ax2.set_ylabel('Height (m)')
    
    plt.tight_layout()
    return fig

# Example usage
def run_simulation():
    # Create structures
    tetrapod_barrier = TetrapodBarrier(length=100, rows=5)
    levee_system = LeveeSystem(length=1000, sections=20)
    
    # Evaluate protection levels
    wave_height = 2.5  # meters
    water_level = 4.0  # meters
    
    barrier_protection = tetrapod_barrier.evaluate_protection(wave_height)
    levee_min_safety, levee_safety_factors = levee_system.evaluate_flood_protection(water_level)
    
    print(f"Tetrapod Barrier Protection Factor: {barrier_protection:.2f}")
    print(f"Levee System Minimum Safety Factor: {levee_min_safety:.2f}")
    
    # Visualize
    fig = visualize_structures(tetrapod_barrier, levee_system)
    return fig

if __name__ == "__main__":
    run_simulation()
