import time
import RPi.GPIO as GPIO
            
class MotorController():
    def __init__(self, io_pin: int = 26):
        self.pin = io_pin
        GPIO.setup(self.pin, GPIO.OUT)
        
    def vibrate(self):
        GPIO.output(self.pin, 1)
    
    def stable(self):
        GPIO.output(self.pin, 0)
        
    def keep_vibrating(self, vibration_time: float = 1):
        self.vibrate()
        time.sleep(vibration_time)
        self.stable()
            
