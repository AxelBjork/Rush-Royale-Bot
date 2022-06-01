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
import bot_core

####
#### Unit type recognition
####

#Feature detection in query image with ORB detector
def feature_match(img_query,img_train):
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img_query,None) # queryImage
        kp2, des2 = orb.detectAndCompute(img_train,None) # trainImage
        if des2 is None: # if no features are found, return
            return 'No Matches'
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        #if len(matches)==0: # if no matches are found, return
        #    return 'No Matches'
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
        current_icons.append([target,match,len(matches)])
    unit_df=pd.DataFrame(current_icons, columns=['icon_'+grid_id,'feature_distance','num_match'])
    if not guess_unit:
        return unit_df
    if guess_unit:
        guess=unit_df.loc[unit_df['feature_distance'].idxmin()] # select empty if all 700
        unit_pred = guess[0]
        return [unit_df.columns[0],unit_pred,guess[1]]

# Get status of current grid
def grid_status(names,prev_grid=None):  # Add multithreading of match unit, match rank??
    grid_stats=[]
    for filename in names:
        rank = rank_error = 0
        unit_guess= match_unit(filename)
        if unit_guess[1]!='empty.png':
            rank,rank_error= match_rank(filename)
        grid_stats.append([*unit_guess,rank,rank_error])
    grid_df=pd.DataFrame(grid_stats, columns=['grid_id','unit','probability','rank','rank_error'])
    # Add grid position 
    box_id=[[(i//5)%5,i%5] for i in range(15)]
    grid_df['position']=box_id
    if not prev_grid is None:
        # Check Consistency
        consistency = grid_df[['grid_id','unit','rank']] ==prev_grid[['grid_id','unit','rank']]
        consistency = consistency.all(axis=1)
        # Update age from previous grid
        grid_df['Age']=prev_grid['Age'] * consistency
        grid_df['Age'] += consistency
    else:
        grid_df['Age']=np.zeros(len(grid_df))
    return grid_df

####
#### Unit rank recognition
####

# show calculated polygon
def show_contour(cnts,shape=(120, 120)):
    canvas = np.zeros(shape, np.uint8)
    img_shape=cv2.drawContours(canvas, [cnts], -1, 255, 3)
    return img_shape

def find_polygon(edges,num=1):
    # Find Contours in image
    cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # Take n largest polygon in image
    #cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:num] # only closed loops
    cnts = sorted(cnts, key=lambda x: cv2.arcLength(x, False), reverse = True)[:num]
    if num==1:
        cnts=cnts[0]
    return cnts

def get_poly(filename,i=4,shape=(120, 120),debug=False):
    # read image as grayscale
    img = cv2.imread(filename,0)
    # Find edges in image
    edges = cv2.Canny(img,100,200)
    # Remove unit in the middle
    keep=35
    edges[keep:-keep,keep:-keep] = 0
    # pick top 3 polygons
    cnts_raw=find_polygon(edges,5)
    if len(cnts_raw)==0:
        return None
    # Redraw Contours
    canvas = np.zeros(shape, np.uint8)
    for cnt in cnts_raw:
        img_cnt=cv2.drawContours(canvas, [cnt], -1, 255, 3)
    # Blur together Contours
    img_cnt = cv2.GaussianBlur(img_cnt,(5,5),0)
    cnts=find_polygon(img_cnt,1)
    # Approximate Polygon
    i=4
    approx=cv2.approxPolyDP(cnts,1.5**i*0.01*cv2.arcLength(cnts,True),True)
    if debug:
        return approx,cnts,img_cnt,cnts_raw,edges,img
    return approx



# Find nearest coordinate in 2D using mean square error
def mean_square(array, coord):
    # take MSE (mean square error)
    coord_mse = ((array-coord)**2).mean(axis=2)
    idx=coord_mse.argmin()
    return coord_mse[idx][0]
# Run mean square of all ranks
# Returns guess of unit rank
def match_rank(filename):
    # Load dictionary with expected corner positions
    corner_dict={
    'rank2':np.array([[10,10],[110,110]]),
    'rank3':np.array([[5,10],[115,10],[60,110]]),
    'rank4':np.array([[60,0],[0,60],[60,120],[120,60]]),
    'rank5':np.array([[15,10],[15,80],[60,115],[105,80],[105,10]]),    
        }
    # Get polygon of image
    target_corners=get_poly(filename)
    if target_corners is None:
       return 0, [0]*len(corner_dict)
    match_errors=[]
    for rank in corner_dict:
        polygon_error = 0
        # Take mean square loss for each corner
        for corner in corner_dict[rank]:
            polygon_error += mean_square(target_corners,corner)
        match_errors.append(polygon_error)
    # Prefer higher ranks 
    match_errors = match_errors *np.array([10,3,2,1])
    np.array(match_errors)
    rank_guess=np.array(match_errors).argmin()
    if match_errors[rank_guess]>1000:
        rank_guess = -1 # rank 1 if none are good match
    return rank_guess+2, match_errors[rank_guess]
    