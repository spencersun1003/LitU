import RPi.GPIO as GPIO

class vibrationSensor():
	def __init__(self,pinA,pinD):
		self.pinA=pinA
		self.pinD=pinD
		
		



if __name__=="__main__":
	
	GPIO.setmode(GPIO.BOARD);
	GPIO.setup(11, GPIO.IN)
	GPIO.setup(13, GPIO.IN)
	while True:
		input_value = GPIO.input(11)
		print("11pin: " + str(input_value),"13pin: "+str(GPIO.input(13)))
		
