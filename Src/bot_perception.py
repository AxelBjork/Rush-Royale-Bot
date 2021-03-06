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
from sklearn.linear_model import LogisticRegression
import pickle
# internal
import bot_core

####
#### Unit type recognition
####

#Feature detection in query image with ORB detector
def feature_match(img_query,img_train):
        # Initiate ORB detector
        orb = cv2.ORB_create()
        orb.setFastThreshold(0)
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img_query,None) # queryImage
        kp2, des2 = orb.detectAndCompute(img_train,None) # trainImage
        if des2 is None: # if no features are found, return
            return 'No Matches'
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        return matches


def match_unit(filename,guess_unit=True):
    # Compare each channel
    current_icons=[]
    img_rgb = cv2.imread(filename)
    # Check every target in dir
    for target in os.listdir("units"):
        # Load icon, 90x90 pixel out of 120 pixel box
        imgSrc=f'units/{target}'
        template_rgb = cv2.imread(imgSrc)
        match=0
        for color in range(3):
            template = template_rgb[:,:,color]
            img_color = img_rgb[:,:,color]
            # Do feature detection
            matches = feature_match(template,img_color)
            # Check matches
            if matches == 'No Matches':
                 return ['empty.png', 999]
            if len(matches)>=10:
                for i in range(10):
                    match+=matches[i].distance
        current_icons.append([target,match,len(matches)])
    unit_df=pd.DataFrame(current_icons, columns=['unit','feature_distance','num_match'])
    if not guess_unit:
        return unit_df
    if guess_unit:
        guess=unit_df.loc[unit_df['feature_distance'].idxmin()] # select empty if all 700
        # Certainty of guess (estimate)
        certainty = round(1 - (guess[1]/unit_df['feature_distance'].sum()),3)
        return [guess[0],certainty]

# Get status of current grid
def grid_status(names,prev_grid=None):  # Add multithreading of match unit, match rank??
    grid_stats=[]
    for filename in names:
        rank,rank_prob= match_rank(filename)
        unit_guess= match_unit(filename) if rank !=0 else ['empty.png',0]
        grid_stats.append([*unit_guess,rank,rank_prob])
    grid_df=pd.DataFrame(grid_stats, columns=['unit','probability','rank','rank_prob'])
    # Add grid position 
    box_id=[[(i//5)%5,i%5] for i in range(15)]
    grid_df.insert(0,'grid_pos',box_id)
    if not prev_grid is None:
        # Check Consistency
        consistency = grid_df[['grid_pos','unit','rank']] ==prev_grid[['grid_pos','unit','rank']]
        consistency = consistency.all(axis=1)
        # Update age from previous grid
        grid_df['Age']=prev_grid['Age'] * consistency
        grid_df['Age'] += consistency
    else:
        grid_df['Age']=np.zeros(len(grid_df))
    return grid_df

def match_rank(filename):
    img = cv2.imread(filename,0)
    edges = cv2.Canny(img,50,100)
    with open('rank_model.pkl', 'rb') as f:
        logreg = pickle.load(f)
        classes = logreg.classes_
    prob = logreg.predict_proba(edges.reshape(1,-1))
    return prob.argmax(),prob.max()

## Add to dataset
def add_grid_to_dataset():
    for slot in os.listdir("OCR_inputs"):
        target=f'OCR_inputs/{slot}'
        img = cv2.imread(target,0)
        edges = cv2.Canny(img,50,100)
        rank_guess  = 0
        unit_guess= match_unit(target)
        if unit_guess[1]!='empty.png':
            rank_guess,_= match_rank(target)
        example_count=len(os.listdir("machine_learning/inputs"))
        cv2.imwrite(f'machine_learning/inputs/{rank_guess}_input_{example_count}.png', edges)
        cv2.imwrite(f'machine_learning/raw_input/{rank_guess}_raw_{example_count}.png', img)

def load_dataset(folder):
    X_train=[]
    Y_train=[]
    for file in os.listdir(folder):
        if file.endswith(".png"):
            X_train.append(cv2.imread(folder+file,0))
            Y_train.append(file.split('_input')[0])
    X_train=np.array(X_train)
    data_shape = X_train.shape
    X_train=X_train.reshape(data_shape[0],data_shape[1]*data_shape[2])
    Y_train=np.array(Y_train, dtype=int)
    return X_train,Y_train

def quick_train_model():
    X_train,Y_train = load_dataset("machine_learning\\inputs\\")
    # train logistic regression model
    logreg =  LogisticRegression()
    logreg.fit(X_train,Y_train)
    return logreg
    