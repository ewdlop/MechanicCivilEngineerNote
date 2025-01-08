import numpy as np
import matplotlib.pyplot as plt

class SpringCouple:
    """
    A metaphorical and mathematical model of coupled oscillators
    representing two intertwined lives during springtime
    """
    
    def __init__(self, 
                 spring_constant_1=100,  # N/m
                 spring_constant_2=100,  # N/m
                 coupling_strength=50,   # Emotional resonance
                 initial_displacement_1=0.1,  # Initial separation
                 initial_displacement_2=0.0):
        """
        Initialize the coupled spring system of love
        """
        self.k1 = spring_constant_1
        self.k2 = spring_constant_2
        self.coupling = coupling_strength
        self.x1 = initial_displacement_1
        self.x2 = initial_displacement_2
    
    def harmonic_motion(self, time_span=10, dt=0.1):
        """
        Simulate the synchronized dance of coupled oscillators
        
        Represents the intertwined lives of two beings
        """
        # Time array
        t = np.arange(0, time_span, dt)
        
        # Initialize displacement arrays
        x1 = np.zeros_like(t)
        x2 = np.zeros_like(t)
        
        # Initial conditions
        x1[0] = self.x1
        x2[0] = self.x2
        
        # Coupled oscillator equations
        for i in range(1, len(t)):
            # Acceleration of first oscillator
            a1 = -(self.k1 * x1[i-1] + 
                   self.coupling * (x1[i-1] - x2[i-1]))
            
            # Acceleration of second oscillator
            a2 = -(self.k2 * x2[i-1] + 
                   self.coupling * (x2[i-1] - x1[i-1]))
            
            # Update velocities and positions
            x1[i] = x1[i-1] + a1 * dt
            x2[i] = x2[i-1] + a2 * dt
        
        return t, x1, x2
    
    def visualize_coupling(self):
        """
        Visualize the synchronized motion
        Like two hearts beating in resonance
        """
        t, x1, x2 = self.harmonic_motion()
        
        plt.figure(figsize=(10, 6))
        plt.plot(t, x1, label='First Oscillator (Love)')
        plt.plot(t, x2, label='Second Oscillator (Harmony)')
        plt.title('Coupled Oscillators: A Spring Romance')
        plt.xlabel('Time')
        plt.ylabel('Displacement')
        plt.legend()
        plt.show()

# Poetic Interpretation
def spring_love_poem():
    """
    A poem capturing the physics of love
    """
    return """
    Coupled Springs of Springtime

    Like two oscillators in perfect phase,
    Our hearts resonate through gentle sways,
    A coupling strength beyond mere physics' might,
    Two souls entwined in harmonic delight.

    The spring constant of our love runs deep,
    Where mathematical precision and emotion meet,
    Displaced by feelings, yet perfectly aligned,
    A differential equation of hearts intertwined.

    In springtime's embrace, we synchronize,
    Our motions coupled beneath blossoming skies,
    Each movement echoes, each vibration shared,
    A quantum entanglement of souls declared.
    """

def main():
    # Create our coupled spring system
    love_oscillators = SpringCouple(
        coupling_strength=75,  # Strong emotional connection
        initial_displacement_1=0.2,  # Initial spark
        initial_displacement_2=0.0   # Responding harmony
    )
    
    # Visualize the coupling
    love_oscillators.visualize_coupling()
    
    # Share the poetic interpretation
    print(spring_love_poem())

if __name__ == "__main__":
    main()
