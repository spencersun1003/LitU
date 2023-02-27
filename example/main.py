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
from DisplayControl import LCD
sys.path.append("..")

lux_history = []
lux_lock = Lock()

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

if __name__=='__main__':
    # lcd=LCD(fps=60)
    # lcd.show_folder('../TestAni')
    procs = []
    lux_reader = LuxReader(5)
    lux_reader.start()
    while True:
        print(lux_history)
        time.sleep(1)
        




