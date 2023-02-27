import RPi.GPIO as GPIO

class vibrationSensor():
	def __init__(self,pinD=22):
		#self.pinA=pinA
		self.pinD=pinD
		GPIO.setup(self.pinD, GPIO.IN)
		
	def read_value(self):
		return GPIO.input(self.pinD)
		
		



if __name__=="__main__":
	reader = vibrationSensor()
	while True:
		print("read value: %d" % reader.read_value())
		
