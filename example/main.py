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
from DisplayControl import LCD, LOADED_IMAGES
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
            change_cnt = 0
            for i in range(len(vibration_history) - 1):
                if vibration_history[i] == 1 and vibration_history[i+1] == 0:
                    change_cnt += 1
                if change_cnt == 5:
                    break
            if change_cnt == 2:
                self.controller.keep_vibrating(0.2)
                time.sleep(0.2)
                self.controller.keep_vibrating(0.2)
            

class DisplayThread(Thread):
    def __init__(self):
        super().__init__()
        self.controller = LCD(fps=30)
        self.timer = 0
        self.imagefps = 30
        self.timeperimage = self.controller.fps / self.imagefps
        self.status = 1
            
    def run(self):
        mode = 'happy'
        file_i = 0
        while True:
            self.controller.show_img(mode, file_i)
            if self.timer > self.timeperimage:
                file_i += 1
                self.timer = 0
            if file_i == len(LOADED_IMAGES[mode]):
                file_i = 0
            self.timer += 1
            if self.status == 1 and np.mean(lux_history) < 50:
                mode = 'cry'
                file_i = 0
                self.status = 2
            elif self.status == 2 and np.mean(lux_history) > 50:
                mode = 'happy'
                file_i = 0
                self.status = 1


if __name__=='__main__':
    procs = []
    lux_reader = LuxReader(5)
    lux_reader.start()
    vibration_reader = VibrationReader(freq=100,limit=25)
    vibration_reader.start()
    motor_thread = MotorControlThread()
    motor_thread.start()
    display_thread = DisplayThread()
    display_thread.start()
    while True:
        print("Lux history read %.4f" % np.mean(lux_history))
        print("Vibration history read %s" % np.array(vibration_history))
        time.sleep(1)
        




