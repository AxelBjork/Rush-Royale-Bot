import os
import time
import numpy as np
import pandas as pd
# Android ADB
from scrcpy import Client, const
import threading

# Image processing
from PIL import Image
import cv2
import pytesseract

SLEEP_DELAY=0.1

class Bot:
    def __init__(self):
        self.device = 'emulator-5554'
        self.screenshotName = self.device + '-screenshot.png'
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
        self.client = Client(device=self.device)
        # Start scrcpy client
        self.client.start(threaded=True)
        time.sleep(0.5)
        # Turn off video stream (spammy)
        self.client.alive=False

    def stop(self):
        self.client.stop()

    # Function to send ADB shell command
    def shell(self, cmd):
        os.system(f'"C:/Programs/Scrcpy/adb" -s {self.device} shell {cmd}')
    # Send ADB to click screen
    def click(self, x, y):
        self.client.control.touch(x, y, const.ACTION_DOWN)
        time.sleep(SLEEP_DELAY/2)
        self.client.control.touch(x, y, const.ACTION_UP)
        time.sleep(SLEEP_DELAY)
    # Click button coords offset and extra delay
    def click_button(self,pos):
        coords=np.array(pos)+10
        self.click(*coords)
        time.sleep(SLEEP_DELAY*10)

    # Send key command, see py-scrcpy consts
    def key_input(self,key):
        self.client.control.keycode(key)
    # Take screenshot of device screen and load pixel values 
    def getScreen(self):
        self.shell(f'/system/bin/screencap -p /sdcard/{self.screenshotName}')
        # Using the adb command to upload the screenshot of the mobile phone to the current directory
        os.system(f'"C:/Programs/Scrcpy/adb" -s {self.device} pull /sdcard/{self.screenshotName}')
    # Crop latest screenshot taken
    def crop_img(self, x, y, dx, dy, name='timeTrain.png'):
        # Load screen
        img_rgb = cv2.imread(self.screenshotName)
        img_rgb = img_rgb[y:y + dy, x:x + dx]
        # Convert to grayscale (done internally by tessarct )
        #img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        #(a, img_gray) = cv2.threshold(img_rgb, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(name, img_rgb)
    # Perform OCR on target area
    def getText(self, x, y, dx, dy,new=True,digits=False):
        if new: self.getScreen()
        # crop image
        self.crop_img(x, y, dx, dy)
        # Do OCR with Google Tesseract
        if digits:
            ocr_text = pytesseract.image_to_string('timeTrain.png',config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789').replace('\n', '')
        else:
            ocr_text = pytesseract.image_to_string('timeTrain.png',config='--psm 13').replace('\n', '')#.replace(' ', '')
        return (ocr_text)
    # find icon on screen
    def getXYByImage(self, target,new=True):
        valid_targets = ['battle_icon','pvp_button','back_button','cont_button','fighting']
        if not target in valid_targets: 
            return "INVALID TARGET" 
        if new: self.getScreen()
        imgSrc=f'icons/{target}.png'
        img_rgb = cv2.imread(f"{self.screenshotName}")
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        template = cv2.imread(imgSrc, 0)
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        threshold = 0.8
        loc = np.where(res >= threshold)
        if len(loc[0]) > 0:
            y = loc[0][0]
            x = loc[1][0]
            return [x, y]
    # Check if any icons are on screen
    def get_current_icons(self,new=True):
        current_icons=[]
        # Update screen and load screenshot as grayscale
        if new: self.getScreen()
        img_rgb = cv2.imread(f"{self.screenshotName}")
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        # Check every target in dir
        for target in os.listdir("icons"):
            x=0 # reset position
            y=0
            # Load icon
            imgSrc=f'icons/{target}'
            template = cv2.imread(imgSrc, 0)
            # Compare images
            res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            threshold = 0.8
            loc = np.where(res >= threshold)
            icon_found = len(loc[0]) > 0
            if icon_found:
                y = loc[0][0]
                x = loc[1][0]
            current_icons.append([target,icon_found,(x,y)])
        icon_df=pd.DataFrame(current_icons, columns=['icon','available','pos [X,Y]'])
        return icon_df
