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

    def merge_unit(self,grid_df,target):
        df_split=grid_df.groupby("unit")
        df_groups=df_split["unit"].count()
        if df_groups[target]>1:
            unit_type_df=df_split.get_group(target).reset_index(drop=True)
            unit_chosen=random.sample(unit_type_df['position'].tolist(), 2)
            self.swipe(*unit_chosen)
            return 'Merged!', unit_type_df
        else: return 'not enough', df_groups
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

#Feature detection in query image with ORB detector
def feature_match(img_query,img_train):
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img_query,None) # queryImage
        kp2, des2 = orb.detectAndCompute(img_train,None) # trainImage
        if des2 is None:
            return 'No Matches'
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        return matches


def match_unit(file_name,guess_unit=True):
    # Compare each channel
    current_icons=[]
    img_gray = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
    # Check every target in dir
    for target in os.listdir("units"):
        grid_id=file_name.split('_')[2].split('.')[0]
        # Load icon, 90x90 pixel out of 120 pixel box
        imgSrc=f'units/{target}'
        template = cv2.imread(imgSrc,cv2.IMREAD_GRAYSCALE)
        match=0
        # Do feature detection
        matches = feature_match(template,img_gray)
        # Check matches
        if matches == 'No Matches':
             return ['icon_'+grid_id,'empty.png', 999]
        if len(matches)>10:
            for i in range(10):
                match+=matches[i].distance
        else:
            match=700
        current_icons.append([target,match])
    unit_df=pd.DataFrame(current_icons, columns=['icon_'+grid_id,'feature_distance'])
    if not guess_unit:
        return unit_df
    if guess_unit:
        guess=unit_df.loc[unit_df['feature_distance'].idxmin()]
        unit_pred = guess[0]
        #if guess[1]>0.55:
        #    unit_pred = 'empty.png'
        return [unit_df.columns[0],unit_pred,guess[1]]

# Get status of current grid
def grid_status(names):
    grid_stats=[]
    for filename in names:
        unit_df= match_unit(filename)
        grid_stats.append(unit_df)
    grid_df=pd.DataFrame(grid_stats, columns=['grid_id','unit','probability'])
    box_id=[[(i//5)%5,i%5] for i in range(15)]
    grid_df['position']=box_id
    return grid_df

def get_unit_count(grid_df):
    df_split=grid_df.groupby("unit")
    df_groups=df_split["unit"].count()
    unit_list=list(df_groups.index)
    return df_split,df_groups, unit_list

####
#### Unit rank recognition
####
# get polygon
def get_poly(file_name,i=4):
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,230,255,cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # Take largest polygon
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    #Estimate corners
    i=4
    approx=cv2.approxPolyDP(cnts,1.5**i*0.01*cv2.arcLength(cnts,True),True)
    return approx

# Find nearest coordinate in 2D using mean square error
def mean_square(array, coord):
    # take MSE (mean square error)
    coord_mse = ((array-coord)**2).mean(axis=2)
    idx=coord_mse.argmin()
    return coord_mse[idx][0]
# Run mean square of all ranks
# Returns guess of unit rank
def match_rank(target_corners):
    corner_dict={
    'rank2':np.array([[10,10],[110,110]]),
    'rank3':np.array([[5,10],[115,10],[60,110]]),
    'rank4':np.array([[60,0],[0,60],[60,120],[120,60]]),
    'rank5':np.array([[15,10],[15,80],[60,115],[105,80],[105,10]]),    
        }
    #figs=[show_contour(corner_dict[rank]) for rank in corner_dict]
    #display(Image.fromarray(np.hstack(figs)))
    match_errors=[]
    for rank in corner_dict:
        polygon_error = 0
        for corner in corner_dict[rank]:
            polygon_error += mean_square(target_corners,corner)
        match_errors.append(polygon_error)
    # Prefer higher ranks 
    match_errors = match_errors *np.array([10,3,2,1])
    np.array(match_errors)
    rank_guess=np.array(match_errors).argmin()
    if match_errors[rank_guess]>1000:
        rank_guess = -1 # rank 1 if none are good match
    return rank_guess+2, match_errors
    