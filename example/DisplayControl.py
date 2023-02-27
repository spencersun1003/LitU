import os
import sys 
import time
import logging
import spidev as SPI
sys.path.append("..")
from lib import LCD_1inch28
from PIL import Image,ImageDraw,ImageFont



class LCD():
    def __init__(self, fps):
        # Raspberry Pi pin configuration:
        RST = 6
        DC = 5
        BL = 13
        bus = 0 
        device = 0
        self.fps = fps
        self.delay = 1 / fps
        self.timer = 0
        self.arcmax = 135
        self.arcspeed = 270 # arc per second
        self.imagefps = 60
        self.timeperimage = self.fps / self.imagefps
        
        logging.basicConfig(level=logging.DEBUG)
        try:
            # display with hardware SPI:
            ''' Warning!!!Don't  creation of multiple displayer objects!!! '''
            self.disp = LCD_1inch28.LCD_1inch28(spi=SPI.SpiDev(bus, device),spi_freq=10000000,rst=RST,dc=DC,bl=BL)
            #self.disp = LCD_1inch28.LCD_1inch28()
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
            
            
    def show_img(self,path='../pic/LCD_1inch28_1.jpg'):
        face = Image.open(path)
        image = Image.new("RGB", (self.disp.width, self.disp.height), "BLACK")	
        face_r=face.rotate(180)
        face_r = face_r.resize((120,120))
        image.paste(face_r, (60,60))
        # Create blank image for drawing.
        draw = ImageDraw.Draw(image)

        # logging.info("draw point")
        # draw.rectangle((20,20,220,220), fill = "blue")
        # logging.info("draw circle")
        draw.arc((10,10,230,230),0, min(self.arcmax, self.delay * self.timer * self.arcspeed), fill =(0,0,255),width=5)
        self.disp.ShowImage(image)
        time.sleep(self.delay)
        self.timer += 1
    def end(self):
        self.disp.module_exit()
        logging.info("quit:")
