import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation

class JointType(Enum):
    """Types of mechanical joints"""
    REVOLUTE = "revolute"          # Rotation only (pin joint)
    PRISMATIC = "prismatic"        # Translation only (sliding joint)
    CYLINDRICAL = "cylindrical"    # Rotation + translation along axis
    SPHERICAL = "spherical"        # 3D rotation (ball joint)
    PLANAR = "planar"             # Translation in plane + rotation
    FIXED = "fixed"               # No relative motion
    UNIVERSAL = "universal"        # Two rotational degrees of freedom

@dataclass
class Joint:
    """Mechanical joint with constraints"""
    type: JointType
    position: np.ndarray
    body1_idx: int
    body2_idx: int
    constraints: Dict[str, float]  # Joint-specific constraints
    angle: float = 0.0
    
    def get_dof(self) -> int:
        """Get degrees of freedom for joint type"""
        dof_map = {
            JointType.REVOLUTE: 1,     # θ
            JointType.PRISMATIC: 1,    # d
            JointType.CYLINDRICAL: 2,  # θ, d
            JointType.SPHERICAL: 3,    # θx, θy, θz
            JointType.PLANAR: 3,       # x, y, θ
            JointType.FIXED: 0,        # None
            JointType.UNIVERSAL: 2     # θ1, θ2
        }
        return dof_map[self.type]

@dataclass
class RigidBody:
    """Rigid body with mass properties"""
    position: np.ndarray
    velocity: np.ndarray
    angular_velocity: float
    mass: float
    inertia: float
    fixed: bool = False

@dataclass
class Pulley:
    """Pulley with radius and position"""
    position: np.ndarray
    radius: float
    angle: float = 0.0
    angular_velocity: float = 0.0
    fixed: bool = True

@dataclass
class Rope:
    """Rope connecting pulleys"""
    pulley1_idx: int
    pulley2_idx: int
    length: float
    tension: float = 0.0

class MechanicalSystem:
    """Physics simulation for joints and pulleys"""
    
    def __init__(self, dt: float = 0.001):
        self.bodies: List[RigidBody] = []
        self.joints: List[Joint] = []
        self.pulleys: List[Pulley] = []
        self.ropes: List[Rope] = []
        self.dt = dt
        self.gravity = np.array([0, -9.81])
        self.time = 0
    
    def add_body(self, position: np.ndarray, mass: float = 1.0,
                 inertia: float = 1.0, fixed: bool = False) -> int:
        """Add rigid body to system"""
        body = RigidBody(
            position=np.array(position),
            velocity=np.zeros(2),
            angular_velocity=0.0,
            mass=mass,
            inertia=inertia,
            fixed=fixed
        )
        self.bodies.append(body)
        return len(self.bodies) - 1
    
    def add_joint(self, type: JointType, position: np.ndarray,
                 body1_idx: int, body2_idx: int,
                 constraints: Dict[str, float] = None) -> None:
        """Add joint between two bodies"""
        if constraints is None:
            constraints = {}
            
        joint = Joint(
            type=type,
            position=np.array(position),
            body1_idx=body1_idx,
            body2_idx=body2_idx,
            constraints=constraints
        )
        self.joints.append(joint)
    
    def add_pulley(self, position: np.ndarray, radius: float,
                   fixed: bool = True) -> int:
        """Add pulley to system"""
        pulley = Pulley(
            position=np.array(position),
            radius=radius,
            fixed=fixed
        )
        self.pulleys.append(pulley)
        return len(self.pulleys) - 1
    
    def add_rope(self, pulley1_idx: int, pulley2_idx: int,
                 length: float) -> None:
        """Connect two pulleys with rope"""
        rope = Rope(
            pulley1_idx=pulley1_idx,
            pulley2_idx=pulley2_idx,
            length=length
        )
        self.ropes.append(rope)
    
    def calculate_mechanical_advantage(self, input_pulley_idx: int,
                                    output_pulley_idx: int) -> float:
        """Calculate ideal mechanical advantage of pulley system"""
        # Simple case: MA = radius_out / radius_in
        r_in = self.pulleys[input_pulley_idx].radius
        r_out = self.pulleys[output_pulley_idx].radius
        
        # Count intermediate pulleys and their arrangement
        intermediate_pulleys = 0
        for rope in self.ropes:
            if (rope.pulley1_idx in [input_pulley_idx, output_pulley_idx] or
                rope.pulley2_idx in [input_pulley_idx, output_pulley_idx]):
                intermediate_pulleys += 1
        
        return (r_out / r_in) * (2 ** (intermediate_pulleys - 1))
    
    def apply_joint_constraints(self) -> None:
        """Apply constraints based on joint types"""
        for joint in self.joints:
            body1 = self.bodies[joint.body1_idx]
            body2 = self.bodies[joint.body2_idx]
            
            if joint.type == JointType.REVOLUTE:
                # Keep bodies at fixed distance, allow rotation
                if not body1.fixed:
                    body1.position = (joint.position - 
                                    joint.constraints.get('radius1', 1.0) *
                                    np.array([np.cos(joint.angle),
                                            np.sin(joint.angle)]))
                if not body2.fixed:
                    body2.position = (joint.position + 
                                    joint.constraints.get('radius2', 1.0) *
                                    np.array([np.cos(joint.angle),
                                            np.sin(joint.angle)]))
            
            elif joint.type == JointType.PRISMATIC:
                # Maintain alignment, allow translation
                axis = np.array([np.cos(joint.constraints.get('axis_angle', 0)),
                               np.sin(joint.constraints.get('axis_angle', 0))])
                if not body1.fixed:
                    body1.position = (joint.position + 
                                    joint.constraints.get('offset1', 0) * axis)
                if not body2.fixed:
                    body2.position = (joint.position + 
                                    joint.constraints.get('offset2', 0) * axis)
    
    def update_pulley_system(self) -> None:
        """Update pulley rotations and rope tensions"""
        for rope in self.ropes:
            pulley1 = self.pulleys[rope.pulley1_idx]
            pulley2 = self.pulleys[rope.pulley2_idx]
            
            # Calculate rope tension
            direction = pulley2.position - pulley1.position
            distance = np.linalg.norm(direction)
            unit_direction = direction / distance
            
            # Update pulley rotations
            if not pulley1.fixed:
                pulley1.angular_velocity += (rope.tension * pulley1.radius / 
                                           (pulley1.radius ** 2)) * self.dt
                pulley1.angle += pulley1.angular_velocity * self.dt
                
            if not pulley2.fixed:
                pulley2.angular_velocity -= (rope.tension * pulley2.radius /
                                           (pulley2.radius ** 2)) * self.dt
                pulley2.angle += pulley2.angular_velocity * self.dt
    
    def step(self) -> None:
        """Advance simulation by one timestep"""
        self.apply_joint_constraints()
        self.update_pulley_system()
        self.time += self.dt

class MechanicalVisualizer:
    """Visualize mechanical system"""
    
    def __init__(self, system: MechanicalSystem):
        self.system = system
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_aspect('equal')
        
        # Initialize visualization elements
        self.joint_points = []
        self.pulley_circles = []
        self.rope_lines = []
        self.setup_visualization()
    
    def setup_visualization(self):
        """Create initial visualization elements"""
        # Draw joints
        for joint in self.system.joints:
            point = self.ax.plot(joint.position[0], joint.position[1], 'ro')[0]
            self.joint_points.append(point)
        
        # Draw pulleys
        for pulley in self.system.pulleys:
            circle = Circle(pulley.position, pulley.radius, fill=False)
            self.ax.add_patch(circle)
            self.pulley_circles.append(circle)
        
        # Draw ropes
        for rope in self.system.ropes:
            p1 = self.system.pulleys[rope.pulley1_idx].position
            p2 = self.system.pulleys[rope.pulley2_idx].position
            line = self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')[0]
            self.rope_lines.append(line)
    
    def update(self, frame):
        """Update visualization"""
        self.system.step()
        
        # Update joint positions
        for joint, point in zip(self.system.joints, self.joint_points):
            point.set_data([joint.position[0]], [joint.position[1]])
        
        # Update pulley rotations
        for pulley, circle in zip(self.system.pulleys, self.pulley_circles):
            circle.center = pulley.position
        
        # Update rope positions
        for rope, line in zip(self.system.ropes, self.rope_lines):
            p1 = self.system.pulleys[rope.pulley1_idx].position
            p2 = self.system.pulleys[rope.pulley2_idx].position
            line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        
        return self.joint_points + self.pulley_circles + self.rope_lines
    
    def animate(self, frames: int = 200):
        """Create animation"""
        anim = FuncAnimation(self.fig, self.update, frames=frames,
                           interval=20, blit=True)
        plt.show()

def example_usage():
    """Demonstrate usage with examples"""
    
    # Create system
    system = MechanicalSystem(dt=0.001)
    
    # Example 1: Simple pulley system
    p1 = system.add_pulley(np.array([0, 2]), 0.5, fixed=True)
    p2 = system.add_pulley(np.array([2, 0]), 0.3, fixed=False)
    system.add_rope(p1, p2, 3.0)
    
    # Example 2: Revolute joint
    b1 = system.add_body(np.array([-2, 1]), fixed=True)
    b2 = system.add_body(np.array([-1, 0]))
    system.add_joint(
        JointType.REVOLUTE,
        np.array([-2, 1]),
        b1, b2,
        {'radius1': 0.5, 'radius2': 0.5}
    )
    
    # Calculate mechanical advantage
    ma = system.calculate_mechanical_advantage(p1, p2)
    print(f"Mechanical Advantage: {ma:.2f}")
    
    # Visualize
    vis = MechanicalVisualizer(system)
    vis.animate()

if __name__ == "__main__":
    example_usage()
