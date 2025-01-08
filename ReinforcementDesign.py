import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

class BarType(Enum):
    """Standard reinforcement bar types"""
    PLAIN = "plain"
    DEFORMED = "deformed"
    EPOXY_COATED = "epoxy_coated"

@dataclass
class RebarProperties:
    """Properties of reinforcement bars"""
    diameter: float  # mm
    area: float     # mm²
    weight: float   # kg/m
    type: BarType

class RebarSizes:
    """Standard rebar sizes and properties"""
    
    STANDARD_SIZES = {
        6: RebarProperties(6, 28.3, 0.222, BarType.PLAIN),
        8: RebarProperties(8, 50.3, 0.395, BarType.DEFORMED),
        10: RebarProperties(10, 78.5, 0.617, BarType.DEFORMED),
        12: RebarProperties(12, 113.1, 0.888, BarType.DEFORMED),
        16: RebarProperties(16, 201.1, 1.578, BarType.DEFORMED),
        20: RebarProperties(20, 314.2, 2.466, BarType.DEFORMED),
        25: RebarProperties(25, 490.9, 3.853, BarType.DEFORMED),
        32: RebarProperties(32, 804.2, 6.313, BarType.DEFORMED),
    }
    
    @classmethod
    def get_properties(cls, diameter: float) -> RebarProperties:
        """Get properties for given bar diameter"""
        if diameter not in cls.STANDARD_SIZES:
            raise ValueError(f"Invalid rebar size: {diameter}mm")
        return cls.STANDARD_SIZES[diameter]

class ConcreteProperties:
    """Concrete material properties"""
    
    def __init__(self, grade: float):
        """
        Initialize concrete properties
        
        Args:
            grade: Characteristic strength (MPa)
        """
        self.fck = grade  # Characteristic compressive strength
        self.gamma_c = 1.5  # Partial safety factor
        self.fcd = self.fck / self.gamma_c  # Design strength
        self.fctm = 0.3 * self.fck**(2/3)  # Mean tensile strength
        self.Ecm = 22 * (self.fck/10)**0.3 * 1000  # Mean elastic modulus

class ReinforcementDesign:
    """Reinforced concrete design calculations"""
    
    def __init__(self, concrete_grade: float, steel_grade: float = 500):
        self.concrete = ConcreteProperties(concrete_grade)
        self.fyk = steel_grade  # Steel characteristic strength
        self.gamma_s = 1.15  # Steel partial safety factor
        self.fyd = self.fyk / self.gamma_s  # Steel design strength
        self.Es = 200000  # Steel elastic modulus (MPa)
    
    def flexural_reinforcement(self, moment: float, width: float, 
                             depth: float, cover: float) -> Tuple[float, List[float]]:
        """
        Calculate required flexural reinforcement
        
        Args:
            moment: Design moment (kNm)
            width: Section width (mm)
            depth: Section depth (mm)
            cover: Concrete cover (mm)
            
        Returns:
            Tuple of (required area, suggested bar configuration)
        """
        # Effective depth
        d = depth - cover
        
        # Design parameters
        fcd = self.concrete.fcd
        fyd = self.fyd
        
        # Moment capacity coefficient
        K = moment * 1e6 / (width * d * d * fcd)
        
        if K > 0.167:  # Compression reinforcement needed
            raise ValueError("Section requires compression reinforcement")
        
        # Lever arm coefficient
        z = d * (0.5 + np.sqrt(0.25 - K/0.9))
        
        # Required tension reinforcement area
        As_req = moment * 1e6 / (fyd * z)
        
        # Suggest bar configuration
        bars = self._suggest_bar_configuration(As_req)
        
        return As_req, bars
    
    def shear_reinforcement(self, shear: float, width: float, 
                          depth: float, tension_steel_ratio: float) -> Tuple[float, float]:
        """
        Calculate required shear reinforcement
        
        Args:
            shear: Design shear force (kN)
            width: Section width (mm)
            depth: Section depth (mm)
            tension_steel_ratio: Ratio of tension reinforcement
            
        Returns:
            Tuple of (required area per meter, spacing)
        """
        # Effective depth
        d = depth
        
        # Design parameters
        fck = self.concrete.fck
        fctm = self.concrete.fctm
        
        # Design concrete shear strength
        VRd_c = 0.12 * (100 * tension_steel_ratio * fck)**(1/3) * width * d / 1000
        
        if shear <= VRd_c:
            # Minimum shear reinforcement only
            Asw_min = 0.08 * np.sqrt(fck) * width / self.fyk
            return Asw_min, min(0.75*d, 600)
        
        # Required shear reinforcement
        z = 0.9 * d
        theta = 45  # Assumed angle of compression strut
        Asw_req = shear * 1000 / (z * self.fyd * np.cos(np.radians(theta)))
        
        # Maximum spacing
        s_max = 0.75 * d
        
        return Asw_req, s_max
    
    def crack_width(self, moment: float, width: float, depth: float,
                   steel_area: float, bar_diameter: float) -> float:
        """
        Calculate crack width
        
        Args:
            moment: Service moment (kNm)
            width: Section width (mm)
            depth: Section depth (mm)
            steel_area: Area of tension reinforcement (mm²)
            bar_diameter: Diameter of bars (mm)
            
        Returns:
            Predicted crack width (mm)
        """
        # Effective depth
        d = depth
        h = depth
        
        # Maximum crack spacing
        c = 25  # Assumed cover
        s = 100  # Assumed bar spacing
        k1 = 0.8  # Bond coefficient
        k2 = 0.5  # Strain distribution
        k3 = 3.4
        k4 = 0.425
        
        sr_max = k3 * c + k1 * k2 * k4 * bar_diameter / self.concrete.fctm
        
        # Steel stress
        x = self._neutral_axis_depth(moment, width, d, steel_area)
        sigma_s = moment * 1e6 * (d - x) / (steel_area * (d - x/3))
        
        # Mean strain
        epsilon_sm = sigma_s/self.Es
        
        # Crack width
        wk = sr_max * epsilon_sm
        
        return wk
    
    def _neutral_axis_depth(self, moment: float, width: float,
                          depth: float, steel_area: float) -> float:
        """Calculate neutral axis depth"""
        d = depth
        alpha_e = self.Es / self.concrete.Ecm
        
        # Quadratic equation coefficients
        a = width/2
        b = steel_area * alpha_e
        c = -steel_area * alpha_e * d
        
        # Solve quadratic equation
        x = (-b + np.sqrt(b*b - 4*a*c))/(2*a)
        
        return x
    
    def _suggest_bar_configuration(self, required_area: float) -> List[float]:
        """Suggest practical bar configuration"""
        available_sizes = sorted(RebarSizes.STANDARD_SIZES.keys(), reverse=True)
        
        # Try different combinations
        best_config = None
        min_excess = float('inf')
        
        for size in available_sizes:
            bar_area = RebarSizes.get_properties(size).area
            n_bars = np.ceil(required_area / bar_area)
            actual_area = n_bars * bar_area
            excess = actual_area - required_area
            
            if 0 <= excess < min_excess:
                min_excess = excess
                best_config = [size] * int(n_bars)
        
        return best_config

def example_usage():
    """Demonstrate reinforced concrete design calculations"""
    
    # Create design object
    design = ReinforcementDesign(concrete_grade=30)
    
    # Example beam
    moment = 200  # kNm
    shear = 150   # kN
    width = 300   # mm
    depth = 600   # mm
    cover = 25    # mm
    
    # Calculate flexural reinforcement
    try:
        area_req, bars = design.flexural_reinforcement(moment, width, depth, cover)
        print(f"\nFlexural reinforcement:")
        print(f"Required area: {area_req:.0f} mm²")
        print(f"Suggested bars: {len(bars)}T{bars[0]}")
        
        # Calculate shear reinforcement
        steel_ratio = area_req / (width * depth)
        asw_req, spacing = design.shear_reinforcement(shear, width, depth, steel_ratio)
        print(f"\nShear reinforcement:")
        print(f"Required area per meter: {asw_req:.0f} mm²/m")
        print(f"Maximum spacing: {spacing:.0f} mm")
        
        # Check crack width
        wk = design.crack_width(moment*0.7, width, depth, area_req, bars[0])
        print(f"\nCrack width:")
        print(f"Predicted crack width: {wk:.3f} mm")
        
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    example_usage()
