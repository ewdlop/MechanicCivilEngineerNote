import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict, Optional

class MaterialType(Enum):
    """Construction material types"""
    WOOD = "wood"
    BRICK = "brick"
    CONCRETE = "concrete"

@dataclass
class MaterialProperties:
    """Material properties for structural design"""
    density: float          # kg/m³
    elastic_modulus: float  # MPa
    compressive_strength: float  # MPa
    tensile_strength: float     # MPa
    thermal_conductivity: float # W/m·K
    cost_per_unit: float       # $/m³
    carbon_footprint: float    # kg CO2/m³
    fire_rating: float         # hours

class MaterialDatabase:
    """Database of material properties"""
    
    MATERIALS = {
        MaterialType.WOOD: MaterialProperties(
            density=500,
            elastic_modulus=11000,
            compressive_strength=40,
            tensile_strength=8,
            thermal_conductivity=0.12,
            cost_per_unit=800,
            carbon_footprint=100,
            fire_rating=0.75
        ),
        MaterialType.BRICK: MaterialProperties(
            density=1800,
            elastic_modulus=14000,
            compressive_strength=20,
            tensile_strength=2,
            thermal_conductivity=0.6,
            cost_per_unit=400,
            carbon_footprint=200,
            fire_rating=4.0
        ),
        MaterialType.CONCRETE: MaterialProperties(
            density=2400,
            elastic_modulus=30000,
            compressive_strength=30,
            tensile_strength=3,
            thermal_conductivity=1.7,
            cost_per_unit=200,
            carbon_footprint=300,
            fire_rating=3.0
        )
    }

class ColumnDesign:
    """Column design for different materials"""
    
    def __init__(self, material_type: MaterialType):
        self.material = MaterialDatabase.MATERIALS[material_type]
        self.type = material_type
        
    def design_column(self, axial_load: float, height: float, 
                     safety_factor: float = 1.5) -> Dict[str, float]:
        """
        Design column based on material type
        
        Args:
            axial_load: Design axial load (kN)
            height: Column height (m)
            safety_factor: Design safety factor
            
        Returns:
            Dictionary with design results
        """
        results = {}
        
        if self.type == MaterialType.CONCRETE:
            results = self._design_concrete_column(axial_load, height, safety_factor)
        elif self.type == MaterialType.BRICK:
            results = self._design_brick_column(axial_load, height, safety_factor)
        else:  # WOOD
            results = self._design_wood_column(axial_load, height, safety_factor)
            
        return results
    
    def _design_concrete_column(self, axial_load: float, height: float,
                              safety_factor: float) -> Dict[str, float]:
        """Design reinforced concrete column"""
        fcd = self.material.compressive_strength / safety_factor
        
        # Initial sizing
        min_area = (axial_load * 1000) / (0.4 * fcd)
        min_dimension = np.sqrt(min_area)
        
        # Round up to practical size
        dimension = np.ceil(min_dimension / 50) * 50
        actual_area = dimension * dimension
        
        # Calculate reinforcement
        rho_min = 0.01  # Minimum reinforcement ratio
        steel_area = actual_area * rho_min
        
        return {
            "dimension": dimension,
            "area": actual_area,
            "reinforcement_area": steel_area,
            "capacity": actual_area * fcd / 1000,  # kN
            "material_cost": actual_area * height * self.material.cost_per_unit / 1e6
        }
    
    def _design_brick_column(self, axial_load: float, height: float,
                           safety_factor: float) -> Dict[str, float]:
        """Design brick column"""
        fcd = self.material.compressive_strength / safety_factor
        
        # Account for mortar joints
        effective_strength = fcd * 0.8
        
        # Initial sizing
        min_area = (axial_load * 1000) / (0.3 * effective_strength)
        
        # Round to brick dimensions (assume 230x110mm bricks)
        width = np.ceil(np.sqrt(min_area) / 230) * 230
        length = width  # Square column for simplicity
        actual_area = width * length
        
        return {
            "width": width,
            "length": length,
            "area": actual_area,
            "capacity": actual_area * effective_strength / 1000,  # kN
            "material_cost": actual_area * height * self.material.cost_per_unit / 1e6
        }
    
    def _design_wood_column(self, axial_load: float, height: float,
                          safety_factor: float) -> Dict[str, float]:
        """Design timber column"""
        fcd = self.material.compressive_strength / safety_factor
        
        # Calculate slenderness factor
        min_dimension = np.sqrt((axial_load * 1000) / (0.3 * fcd))
        slenderness = height * 1000 / min_dimension
        
        # Apply slenderness reduction
        if slenderness > 10:
            fcd *= (1 - (slenderness - 10) / 200)
        
        # Size column
        min_area = (axial_load * 1000) / (0.3 * fcd)
        dimension = np.ceil(np.sqrt(min_area) / 50) * 50
        actual_area = dimension * dimension
        
        return {
            "dimension": dimension,
            "area": actual_area,
            "capacity": actual_area * fcd / 1000,  # kN
            "material_cost": actual_area * height * self.material.cost_per_unit / 1e6
        }

class SlabDesign:
    """Slab design for different materials"""
    
    def __init__(self, material_type: MaterialType):
        self.material = MaterialDatabase.MATERIALS[material_type]
        self.type = material_type
        
    def design_slab(self, span: float, live_load: float, 
                   safety_factor: float = 1.5) -> Dict[str, float]:
        """
        Design slab based on material type
        
        Args:
            span: Clear span (m)
            live_load: Design live load (kN/m²)
            safety_factor: Design safety factor
            
        Returns:
            Dictionary with design results
        """
        results = {}
        
        if self.type == MaterialType.CONCRETE:
            results = self._design_concrete_slab(span, live_load, safety_factor)
        elif self.type == MaterialType.WOOD:
            results = self._design_wood_slab(span, live_load, safety_factor)
        else:
            raise ValueError("Material not suitable for slab design")
            
        return results
    
    def _design_concrete_slab(self, span: float, live_load: float,
                            safety_factor: float) -> Dict[str, float]:
        """Design reinforced concrete slab"""
        fcd = self.material.compressive_strength / safety_factor
        
        # Estimate depth
        min_depth = span * 1000 / 25  # Basic span/depth ratio
        depth = np.ceil(min_depth / 25) * 25
        
        # Calculate loading
        dead_load = depth * self.material.density / 1000 * 9.81 / 1000  # kN/m²
        total_load = (dead_load + live_load) * safety_factor
        
        # Calculate moment
        moment = total_load * span * span / 8  # kNm/m
        
        # Calculate reinforcement
        d = depth - 25  # Effective depth
        k = moment * 1e6 / (1000 * d * d * fcd)
        if k > 0.156:
            raise ValueError("Depth insufficient for moment")
            
        z = d * (0.5 + np.sqrt(0.25 - k/0.9))
        steel_area = moment * 1e6 / (500 * z)  # Assume 500MPa steel
        
        return {
            "depth": depth,
            "reinforcement_area": steel_area,
            "dead_load": dead_load,
            "material_cost": depth/1000 * 1 * span * self.material.cost_per_unit
        }
    
    def _design_wood_slab(self, span: float, live_load: float,
                         safety_factor: float) -> Dict[str, float]:
        """Design timber slab/floor"""
        fcd = self.material.compressive_strength / safety_factor
        
        # Calculate required section modulus
        total_load = live_load * safety_factor
        moment = total_load * span * span / 8
        required_modulus = moment * 1e6 / (0.6 * fcd)
        
        # Size joists (assume 300mm spacing)
        spacing = 300
        depth = np.ceil(np.sqrt(6 * required_modulus / spacing) / 25) * 25
        width = np.ceil(depth/6 / 25) * 25
        
        return {
            "joist_depth": depth,
            "joist_width": width,
            "joist_spacing": spacing,
            "material_cost": (depth/1000 * width/1000 * span * 
                            (1000/spacing) * self.material.cost_per_unit)
        }

def compare_materials(span: float, load: float, height: float) -> None:
    """Compare different materials for given requirements"""
    
    print("\nMaterial Comparison Analysis:")
    print("=" * 50)
    
    # Column comparison
    print("\nColumn Design Comparison:")
    for material in MaterialType:
        try:
            designer = ColumnDesign(material)
            results = designer.design_column(load, height)
            print(f"\n{material.value.capitalize()}:")
            for key, value in results.items():
                print(f"  {key}: {value:.2f}")
        except ValueError as e:
            print(f"\n{material.value.capitalize()}: Not suitable - {e}")
    
    # Slab comparison
    print("\nSlab Design Comparison:")
    for material in [MaterialType.CONCRETE, MaterialType.WOOD]:
        try:
            designer = SlabDesign(material)
            results = designer.design_slab(span, load)
            print(f"\n{material.value.capitalize()}:")
            for key, value in results.items():
                print(f"  {key}: {value:.2f}")
        except ValueError as e:
            print(f"\n{material.value.capitalize()}: Not suitable - {e}")
    
    # Material properties comparison
    print("\nMaterial Properties Comparison:")
    properties = ["density", "elastic_modulus", "compressive_strength",
                 "thermal_conductivity", "cost_per_unit", "carbon_footprint",
                 "fire_rating"]
    
    for prop in properties:
        print(f"\n{prop.replace('_', ' ').capitalize()}:")
        for material in MaterialType:
            value = getattr(MaterialDatabase.MATERIALS[material], prop)
            print(f"  {material.value.capitalize()}: {value}")

def example_usage():
    """Demonstrate material comparison"""
    
    # Example design requirements
    span = 5.0    # meters
    load = 500.0  # kN
    height = 3.0  # meters
    
    compare_materials(span, load, height)

if __name__ == "__main__":
    example_usage()
