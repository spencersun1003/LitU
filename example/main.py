#!/usr/bin/python
# -*- coding: UTF-8 -*-
#import chardet
import os
import sys 
import time
import logging
from DisplayControl import LCD
sys.path.append("..")



if __name__=='__main__':
    lcd=LCD(fps=60)
    lcd.show_folder('../TestAni')
   #  while True:
   #      lcd.show_img()
        



