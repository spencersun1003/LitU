'''
        Read Gyro and Accelerometer by Interfacing Raspberry Pi with MPU6050 using Python
	http://www.electronicwings.com
'''
import smbus					#import SMBus module of I2C
from time import sleep          #import

#some MPU6050 Registers and their Address
PWR_MGMT_1   = 0x6B
SMPLRT_DIV   = 0x19
CONFIG       = 0x1A
GYRO_CONFIG  = 0x1B
INT_ENABLE   = 0x38
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H  = 0x43
GYRO_YOUT_H  = 0x45
GYRO_ZOUT_H  = 0x47


class IMUsensor():
	def __init__(self):
		#self.pinA=pinA
		
		self.bus = smbus.SMBus(1) 	# or bus = smbus.SMBus(0) for older version boards
		self.Device_Address = 0x68   # MPU6050 device address

		self.MPU_Init()

	def MPU_Init(self):
		#write to sample rate register
		self.bus.write_byte_data(self.Device_Address, SMPLRT_DIV, 7)
		
		#Write to power management register
		self.bus.write_byte_data(self.Device_Address, PWR_MGMT_1, 1)
		
		#Write to Configuration register
		self.bus.write_byte_data(self.Device_Address, CONFIG, 0)
		
		#Write to Gyro configuration register
		self.bus.write_byte_data(self.Device_Address, GYRO_CONFIG, 24)
		
		#Write to interrupt enable register
		self.bus.write_byte_data(self.Device_Address, INT_ENABLE, 1)

	def read_raw_data(self,addr):
		#Accelero and Gyro value are 16-bit
		high = self.bus.read_byte_data(self.Device_Address, addr)
		low = self.bus.read_byte_data(self.Device_Address, addr+1)

		#concatenate higher and lower value
		value = ((high << 8) | low)

		#to get signed value from mpu6050
		if(value > 32768):
			value = value - 65536
		return value
		
	def read_value(self):
		
		#Read Accelerometer raw value
		acc_x = self.read_raw_data(ACCEL_XOUT_H)
		acc_y = self.read_raw_data(ACCEL_YOUT_H)
		acc_z = self.read_raw_data(ACCEL_ZOUT_H)
		
		#Read Gyroscope raw value
		gyro_x = self.read_raw_data(GYRO_XOUT_H)
		gyro_y = self.read_raw_data(GYRO_YOUT_H)
		gyro_z = self.read_raw_data(GYRO_ZOUT_H)
		
		#Full scale range +/- 250 degree/C as per sensitivity scale factor
		Ax = acc_x/16384.0
		Ay = acc_y/16384.0
		Az = acc_z/16384.0
		
		Gx = gyro_x/131.0
		Gy = gyro_y/131.0
		Gz = gyro_z/131.0

		
		return (Ax,Ay,Az),(Gx,Gy,Gz)



if __name__=="__main__":


	print (" Reading Data of Gyroscope and Accelerometer")
	imu=IMUsensor()
	while True:
		
		

		print(imu.read_value())	
		sleep(1)
