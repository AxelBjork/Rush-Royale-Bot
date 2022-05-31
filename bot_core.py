import os
import time
import numpy as np
import pandas as pd
import random
# Android ADB
from scrcpy import Client, const
import threading
# Image processing
from PIL import Image
import cv2
import pytesseract
# internal
import bot_perception


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
    def click(self, x, y,delay_mult=1):
        self.client.control.touch(x, y, const.ACTION_DOWN)
        time.sleep(SLEEP_DELAY/2*delay_mult)
        self.client.control.touch(x, y, const.ACTION_UP)
        time.sleep(SLEEP_DELAY*delay_mult)
    # Click button coords offset and extra delay
    def click_button(self,pos):
        coords=np.array(pos)+10
        self.click(*coords)
        time.sleep(SLEEP_DELAY*10)
    # Swipe on combat grid to merge units
    def swipe(self,start,end):
        boxes,box_size = get_grid()
        # Offset from box edge
        offset = 60
        self.client.control.swipe(*boxes[start[0],start[1]]+offset,
        *boxes[end[0],end[1]]+offset,
        20,1/60)
    # Send key command, see py-scrcpy consts
    def key_input(self,key):
        self.client.control.keycode(key)
    # Kill the game and restart
    def restart_RR(self):
            self.shell(f'input keyevent {const.KEYCODE_APP_SWITCH}') #Go app switch
            time.sleep(1)
            self.click(781,80)# Clear all apps
            time.sleep(2)
            self.shell(f'input tap 420 350') # re-open RR
            time.sleep(10)
    # Take screenshot of device screen and load pixel values 
    def getScreen(self):
        self.shell(f'/system/bin/screencap -p /sdcard/{self.screenshotName}')
        # Using the adb command to upload the screenshot of the mobile phone to the current directory
        os.system(f'"C:/Programs/Scrcpy/adb" -s {self.device} pull /sdcard/{self.screenshotName}')
    # Crop latest screenshot taken
    def crop_img(self, x, y, dx, dy, name='icon.png'):
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
            ocr_text = pytesseract.image_to_string('icon.png',config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789').replace('\n', '')
        else:
            ocr_text = pytesseract.image_to_string('icon.png',config='--psm 13').replace('\n', '')#.replace(' ', '')
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
    # Scan battle grid, update OCR images
    def scan_grid(self,new=False):
        boxes,box_size = get_grid()
        # should be enabled by default
        if new: self.getScreen()
        box_list = boxes.reshape(15,2)
        names=[]
        for i in range(len(box_list)):
            file_name = f'OCR_inputs/icon_{str(i)}.png'
            self.crop_img(*box_list[i],*box_size,name=file_name)
            names.append(file_name)
        return names

    def merge_unit(self,df_split,merge_series):
        # Pick a random filtered target
        merge_target =  merge_series.sample().index[0]
        # Collect unit dataframe
        merge_df=df_split.get_group(merge_target).sample(n=2)
        # Extract unit position from dataframe
        unit_chosen=merge_df['position'].tolist()
        # Send Merge 
        self.swipe(*unit_chosen)
        time.sleep(0.2)
        return merge_df
####
#### END OF CLASS
####

# Get fight grid pixel values
def get_grid():
    #Grid dimensions
    top_box=(153,945)
    box_size=(120,120)
    gap=0
    height=3
    width = 5
    # x_cords
    x_cord=list(range(top_box[0],top_box[0]+(box_size[0]+gap)*width,box_size[0]+gap))
    y_cord=list(range(top_box[1],top_box[1]+(box_size[1]+gap)*height,box_size[1]+gap))
    boxes=[]
    # Create list of all boxes
    for y_point in y_cord:
        for x_point in x_cord:
            boxes.append((x_point,y_point))
    # Convert to np array (4x4) with x,y coords
    boxes = np.array(boxes).reshape(height,width,2)
    return boxes, box_size

def get_unit_count(grid_df):
    df_split=grid_df.groupby("unit")
    df_groups=df_split["unit"].count()
    unit_list=list(df_groups.index)
    return df_split,df_groups, unit_list

# Split grid df into unique units and ranks
# Shows total count of unit and count of each rank
def grid_meta_info(grid_df):
    # Split by unique unit
    df_groups=get_unit_count(grid_df)[1]
    df_split=grid_df.groupby(['unit','rank'])
    # Count number of unit of each rank
    unit_series=df_split['unit'].count()
    unit_series=unit_series.sort_values(ascending=False)
    group_keys = list(unit_series.index)
    return df_split,unit_series, df_groups, group_keys

# Returns all elements which match tokens value
def filter_keys(unit_series,tokens):
    series= []
    for token in tokens:
        # check if given token is int, assume unit rank filter
        if isinstance(token,int):
            exists = unit_series.index.get_level_values('rank').isin([token]).any()
            series.append(unit_series.xs(token, level='rank',drop_level=False) if exists else pd.Series(dtype=object))
        else:
            series.append(unit_series.xs(token, level='unit',drop_level=False) if token in unit_series else pd.Series(dtype=object))
    return pd.concat(series)