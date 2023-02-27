#!/usr/bin/python
# -*- coding: UTF-8 -*-
#import chardet
import os
import sys 
import time
import logging
import multiprocessing
import numpy as np

from threading import Thread, Lock
from LuxSensor import readLight
from VibraionSensor import vibrationSensor
from DisplayControl import LCD
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
            

if __name__=='__main__':
    # lcd=LCD(fps=60)
    # lcd.show_folder('../TestAni')
    procs = []
    lux_reader = LuxReader(5)
    lux_reader.start()
    vibration_reader = VibrationReader(5)
    vibration_reader.start()
    while True:
        print("Lux history read %.4f" % np.mean(lux_history))
        print("Vibration history read %s" % np.array(vibration_history))
        time.sleep(1)
        




