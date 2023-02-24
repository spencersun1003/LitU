#!/usr/bin/python
# -*- coding: UTF-8 -*-
#import chardet
import os
import sys 
import time
import logging
import spidev as SPI
sys.path.append("..")
from lib import LCD_1inch28
from PIL import Image,ImageDraw,ImageFont



class LCD():
    def __init__(self):
        # Raspberry Pi pin configuration:
        RST = 27
        DC = 25
        BL = 21
        bus = 0 
        device = 0
        logging.basicConfig(level=logging.DEBUG)
        try:
            # display with hardware SPI:
            ''' Warning!!!Don't  creation of multiple displayer objects!!! '''
            #disp = LCD_1inch28.LCD_1inch28(spi=SPI.SpiDev(bus, device),spi_freq=10000000,rst=RST,dc=DC,bl=BL)
            self.disp = LCD_1inch28.LCD_1inch28()
            # Initialize library.
            self.disp.Init()
            # Clear display.
            self.disp.clear()
        except IOError as e:
            logging.info(e)    
        except KeyboardInterrupt:
            self.disp.module_exit()
            logging.info("quit:")
            exit()
            
            
    def show_img(self,path='../pic/LCD_1inch28_1.jpg',delay=0.05):
        image = Image.open(path)	
        im_r=image.rotate(180)
        self.disp.ShowImage(im_r)
        time.sleep(delay)
    def end(self):
        self.disp.module_exit()
        logging.info("quit:")
    def show_folder(self,path):
        fileName=os.listdir(path)
        
        while True:
            for file in fileName:
                self.show_img(os.path.join(path,file))


if __name__=='__main__':
    

    lcd=LCD()
    lcd.show_folder('../TestAni')
   #  while True:
   #      lcd.show_img()
        



