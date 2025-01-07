#!/usr/bin/env python3

from ev3dev2.motor import LargeMotor, MoveTank, OUTPUT_A, OUTPUT_B, OUTPUT_C
from ev3dev2.sensor import INPUT_1, INPUT_2, INPUT_3
from ev3dev2.sensor.lego import TouchSensor, ColorSensor, UltrasonicSensor
from ev3dev2.sound import Sound
from ev3dev2.led import Leds
import time

# Initialize our robot's components
tank_drive = MoveTank(OUTPUT_A, OUTPUT_B)
claw_motor = LargeMotor(OUTPUT_C)
touch_sensor = TouchSensor(INPUT_1)
color_sensor = ColorSensor(INPUT_2)
ultrasonic = UltrasonicSensor(INPUT_3)
sound = Sound()
leds = Leds()

def initialize_robot():
    """Reset robot to starting position"""
    sound.speak("Initializing systems")
    leds.set_color("LEFT", "GREEN")
    leds.set_color("RIGHT", "GREEN")
    # Reset claw to open position
    claw_motor.on_for_rotations(speed=20, rotations=-0.5)

def detect_and_grab_object():
    """Find and grab colored objects"""
    while True:
        # Use ultrasonic to find objects
        distance = ultrasonic.distance_centimeters
        
        if distance < 20:  # Object detected within 20cm
            # Check object color
            color = color_sensor.color_name
            
            if color == 'RED':
                sound.speak("Red block detected")
                leds.set_color("LEFT", "RED")
                # Move forward slowly
                tank_drive.on_for_rotations(20, 20, 0.5)
                # Close claw
                claw_motor.on_for_rotations(speed=20, rotations=0.5)
                # Lift object
                tank_drive.on_for_rotations(-20, -20, 0.5)
                break
                
        # If no object found, keep searching
        tank_drive.on_for_degrees(30, -30, 45)  # Turn to scan
        time.sleep(0.1)

def delivery_sequence():
    """Deliver grabbed object to target zone"""
    # Turn to delivery zone
    tank_drive.on_for_degrees(45, -45, 90)
    
    # Move to delivery zone
    tank_drive.on_for_rotations(30, 30, 2)
    
    # Release object
    claw_motor.on_for_rotations(speed=20, rotations=-0.5)
    sound.speak("Package delivered")
    leds.set_color("LEFT", "GREEN")
    
    # Back up
    tank_drive.on_for_rotations(-30, -30, 1)

def dance_celebration():
    """Victory dance after successful delivery"""
    sound.play_song((
        ('C4', 'q'),
        ('E4', 'q'),
        ('G4', 'h')
    ))
    
    # Spin dance
    for _ in range(2):
        tank_drive.on_for_degrees(50, -50, 360)
        tank_drive.on_for_degrees(-50, 50, 360)
    
    leds.animate_flash('GREEN', sleeptime=0.5, duration=3)

def main():
    try:
        initialize_robot()
        while not touch_sensor.is_pressed:  # Wait for start button
            time.sleep(0.1)
            
        sound.speak("Mission starting")
        
        while True:
            detect_and_grab_object()
            delivery_sequence()
            dance_celebration()
            
            # Wait for touch sensor to start next round
            sound.speak("Press button for next mission")
            while not touch_sensor.is_pressed:
                time.sleep(0.1)
                
    except Exception as e:
        sound.speak("Error detected")
        print(e)
    
    finally:
        # Clean shutdown
        tank_drive.off()
        claw_motor.off()
        leds.all_off()

if __name__ == "__main__":
    main()
