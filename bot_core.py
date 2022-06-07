import os
import time
import numpy as np
import pandas as pd
import random
from tqdm.notebook import trange, tqdm
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
        # Try to launch application through ADB shell
        self.shell('monkey -p com.my.defense 1')
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
    # Force restart the game through ADC, or spam 10 disconnects to abandon match
    def restart_RR(self,quick_disconnect=False):
            if quick_disconnect:
                for i in range(40):
                    self.shell(f'input keyevent {const.KEYCODE_APP_SWITCH}') #Go app switch
                    return
            # Force kill game through ADB shell
            self.shell('am force-stop com.my.defense')
            time.sleep(2)
            # Launch application through ADB shell
            self.shell('monkey -p com.my.defense 1')
            time.sleep(10) # wait for app to load
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
    def get_current_icons(self,new=True,available=False):
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
        # filter out only available buttons
        if available:
            icon_df=icon_df[icon_df['available']==True].reset_index(drop=True)
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
    # Take random unit in series, find corresponding dataframe and merge two random ones
    def merge_unit(self,df_split,merge_series):
        # Pick a random filtered target
        if len(merge_series)>0:
            merge_target =  merge_series.sample().index[0]
        else: 
            return merge_series
        # Collect unit dataframe
        merge_df=df_split.get_group(merge_target)
        if len(merge_df)>1:
            merge_df=merge_df.sample(n=2)
        else: 
            return merge_df
        # Extract unit position from dataframe
        unit_chosen=merge_df['position'].tolist()
        # Send Merge 
        self.swipe(*unit_chosen)
        time.sleep(0.2)
        return merge_df
    # Try to find a merge target and merge it
    def try_merge(self,rank=1,prev_grid=None):
        info=''
        merge_df =None
        names=self.scan_grid(new=True)
        grid_df=bot_perception.grid_status(names,prev_grid=prev_grid)
        df_split,unit_series, df_groups, group_keys=grid_meta_info(grid_df)
        # Select stuff to merge
        # Find highest chemist rank
        merge_series = unit_series
        merge_chemist = adv_filter_keys(unit_series,'chemist.png',remove=False)
        if not merge_chemist.empty:
            max_chemist = merge_chemist.index.max()
            # Remove 1 count of highest rank chemist
            merge_series[merge_series.index == max_chemist] = merge_series[merge_series.index == max_chemist] - 1
        # Select stuff to merge
        merge_series = merge_series[merge_series>=2] # At least 2 units ## ADD MIME to every count, use sample rank and check if mime exist
        # check if grid full
        if ('empty.png',0) in group_keys:
            # Try to merge high priority units
            merge_prio = adv_filter_keys(merge_series,[['chemist.png','monkey.png']])
            if not merge_prio.empty:
                merge_df = self.merge_unit(df_split,merge_prio)
            # Merge if full board
            if df_groups['empty.png']<=2:
                info='Merging!'
                # Add criteria
                merge_series = adv_filter_keys(merge_series,rank,remove=False)
                if not merge_series.empty:
                    merge_df = self.merge_unit(df_split,merge_series)
                else:
                    info='not enough rank 1 targets!'
            else: info= 'need more units!'
        # If grid seems full, merge more units
        else:
            info = 'Full Grid - Merging!'
            # Remove all high level crystals and zealots
            merge_series = adv_filter_keys(merge_series,[[3,4,5],['zealot.png','crystal.png']],remove=True)
            if not merge_series.empty:
                merge_df = self.merge_unit(df_split,merge_series)
        return grid_df,unit_series,merge_df,info
    # Mana level cards
    def mana_level(self,cards, hero_power=False):
        upgrade_pos_dict={
        1:[100,1500], 2:[200,1500],
        3:[350,1500], 4:[500,1500],
        5:[650,1500] }
        # Level each card
        for card in cards:
            self.click(*upgrade_pos_dict[card])
        if hero_power:
            self.click(800,1500)

    # Start a dungeon floor from PvE page
    def play_dungeon(self,floor=5):
        print('Starting Dungeon floor', floor)
        # Divide by 3 and take ceiling of floor as int
        target_chapter = f'chapter_{int(np.ceil(floor/3))}.png'
        expanded=0
        pos = np.array([0,0])
        avail_buttons = self.get_current_icons(available=True)
        # Check if on dungeon page
        if (avail_buttons == 'dungeon_page.png').any(axis=None):
            # Swipe to the top
            [self.swipe([0,0],[2,0]) for i in range(10)]
            self.click(30,600,5) # stop scroll and scan screen for buttons
            # Keep swiping until floor is found
            for i in range(10):
                avail_buttons = self.get_current_icons(available=True)
                # Look for correct chapter
                if (avail_buttons == target_chapter).any(axis=None):
                    pos = get_button_pos(avail_buttons,target_chapter)
                    if not expanded:
                        expanded = 1
                        self.click_button(pos+[500,90])
                    # check button is near top of screen
                    if pos[1] < 550:
                        # Stop scrolling
                        break
                # Swipe down (change to swipe up for floor 10 cleared)
                [self.swipe([2,0],[0,0]) for i in range(1)]
                self.click(30,600) # stop scroll and scan screen for buttons
            ## Click play floor if found
            if not (pos == np.array([0,0])).any():
                self.click_button(pos+[0,85+400*(floor%3)]) #(only 1,2, 4,5, 7, 8 possible)
                self.click_button((130,950))
                time.sleep(2) # wait for matchmaking        

    # Locate game home screen and try to start fight is chosen
    def battle_screen(self,start=False,pve=True,floor=5):
        # Scan screen for any key buttons
        df = self.get_current_icons(available=True)
        if not df.empty:
            # list of buttons
            if (df == 'fighting.png').any(axis=None) and not (df == '0cont_button.png').any(axis=None) in df:
                return df,'fighting'
            # Start pvp if homescreen   
            if start and (df == 'pvp_button.png').any(axis=None) and (df == 'battle_icon.png').any(axis=None):
                pvp_pos = get_button_pos(df,'pvp_button.png')
                if pve:
                    # Add a 500 pixel offset for PvE button
                    self.click_button(pvp_pos+[500,0])
                    self.play_dungeon(floor=floor)
                else: self.click_button(pvp_pos)
                time.sleep(1)
                return df, 'home'
            # Check first button is clickable
            df_click = df[df['icon'].isin(['back_button.png','battle_icon.png','0cont_button.png','1quit.png'])]
            if not df_click.empty:
                button_pos=df_click['pos [X,Y]'].tolist()[0]
                self.click_button(button_pos)
                return df,'menu'
        self.shell(f'input keyevent {const.KEYCODE_BACK}') #Force back
        return df,'lost'
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

# Returns all elements which match tokens value with multiple levels
# Either provide nested list of list, simple list or unit type/unit rank and if remove or not
# Criteria example:  [[3,4,5],['zealot.png','crystal.png']]
def adv_filter_keys(unit_series,tokens,remove=False):
    if unit_series.empty:
        return pd.Series(dtype=object)
    if not isinstance(tokens, list): # Make token a list if not already
        tokens = [tokens]
    # Add detection of dimension in input tokens
    merge_series= unit_series
    for level in tokens:
        if not isinstance(level, list): # Make token a list if not already
            level = [level]
        series= []
        for token in level:
            # check if given token is int, assume unit rank filter
            if isinstance(token,int):
                exists = merge_series.index.get_level_values('rank').isin([token]).any()
                if exists:
                    series.append(merge_series.xs(token, level='rank',drop_level=False))
                else: continue # skip if nothing matches criteria
            elif isinstance(token,str):
                if token in merge_series: 
                    series.append(merge_series.xs(token, level='unit',drop_level=False))
                else: continue
        if not len(series)==0:
            merge_series = pd.concat(series)
            if remove:
                merge_series = unit_series[~unit_series.index.isin(merge_series.index)]
            return merge_series
        elif remove: # return empty list if empty and nothing matches criteria
            return pd.Series(dtype=object)
        # Otherwise Do next loop with unchanged merge series otherwise
    # Return result of all criterias
    return merge_series

# Will spam read all knowledge in knowledge base for free gold, roughly 3k, 100 gems
def read_knowledge(bot):
    spam_click=trange(1000)
    for i in spam_click:
        bot.click(450,1300,0.1)

def get_button_pos(df,button):
    #button=button+'.png'
    pos=df[df['icon']==button]['pos [X,Y]'].reset_index(drop=True)[0]
    return np.array(pos)