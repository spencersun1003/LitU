#!/usr/bin/python
# -*- coding: UTF-8 -*-
#import chardet
import os
import sys 
import time
import logging
import multiprocessing
import numpy as np

import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM);
from threading import Thread, Lock
from LuxSensor import readLight
from VibraionSensor import vibrationSensor
from DisplayControl import LCD
from motorcontrol import MotorController
sys.path.append("..")

lux_history = []
lux_lock = Lock()
vibration_history = []
vibration_lock = Lock()

class LuxReader(Thread):
    def __init__(self, freq: float, limit: int = 10):
        super().__init__()
        self.keep_reading = True
        self.reading_freq = freq
        self.reading_delay = 1 / freq
        self.history_limit = limit
        
    def read_light(self):
        return readLight()
    
    def run(self):
        while self.keep_reading:
            lux_lock.acquire()
            if len(lux_history) == self.history_limit:
                lux_history.pop(0)
            lux_history.append(self.read_light())
            lux_lock.release()
            time.sleep(self.reading_delay)
            
class VibrationReader(Thread):
    def __init__(self, freq: float, limit: int = 10):
        super().__init__()
        self.keep_reading = True
        self.reading_freq = freq
        self.reading_delay = 1 / freq
        self.history_limit = limit
        self.reader = vibrationSensor()
        
    def read_vibration(self):
        return self.reader.read_value()
    
    def run(self):
        while self.keep_reading:
            vibration_lock.acquire()
            if len(vibration_history) == self.history_limit:
                vibration_history.pop(0)
            vibration_history.append(self.read_vibration())
            vibration_lock.release()
            time.sleep(self.reading_delay)
            
class MotorControlThread(Thread):
    def __init__(self):
        super().__init__()
        self.controller = MotorController()
    
    def run(self):
        while True:
            time.sleep(3)
            self.controller.keep_vibrating(0.2)
            time.sleep(0.2)
            self.controller.keep_vibrating(0.2)
            

class DisplayThread(Thread):
    def __init__(self):
        super().__init__()
        self.controller = LCD(fps=30)
    
    def run(self):
        self.show_folder('../TestAni')
            
    def show_folder(self,path):
        fileName=os.listdir(path)
        file_i = 0
        while True:
            self.controller.show_img(os.path.join(path,fileName[file_i]))
            if not self.controller.timer % self.controller.timeperimage:
                file_i += 1
            if file_i == len(fileName):
                file_i = 0


if __name__=='__main__':
    procs = []
    lux_reader = LuxReader(5)
    lux_reader.start()
    vibration_reader = VibrationReader(5)
    vibration_reader.start()
    motor_thread = MotorControlThread()
    motor_thread.start()
    display_thread = DisplayThread()
    display_thread.start()
    while True:
        print("Lux history read %.4f" % np.mean(lux_history))
        print("Vibration history read %s" % np.array(vibration_history))
        time.sleep(1)
        




