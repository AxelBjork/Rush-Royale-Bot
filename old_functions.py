# Battle screen from idle state
def battle_screen(start=False):
    #Check if fight
    pos=bot.getXYByImage('fighting')
    if pos!=None:
        return
    #Check back button 
    pos=bot.getXYByImage('0cont_button',new=False)
    if pos!=None:
        for i in range(3):
            bot.click_button(pos)
            time.sleep(4)
    #Check back button 
    pos=bot.getXYByImage('back_button')
    if pos!=None:
        bot.click_button(pos)
    pos=bot.getXYByImage('battle_icon')
    if pos == None:
        bot.key_input(const.KEYCODE_BACK)
        bot.click(460,1580)
        bot.click(460,1580)
        print("fail")
        time.sleep(SLEEP_DELAY*2)
        pos=bot.getXYByImage('battle_icon')
        if pos == None:
            print('fail 2x')
    
    pvp_pos=bot.getXYByImage('pvp_button')
    if start and pvp_pos!=None:
        bot.click_button(pvp_pos)
    return pos,pvp_pos


#Match unit in grid
def match_unit(file_name,guess_unit=True):
    # Compare each channel
    current_icons=[]
    img_rgb = cv2.imread(file_name)
    # Check every target in dir
    for target in os.listdir("units"):
        grid_id=file_name.split('_')[2].split('.')[0]
        x=0 # reset position
        y=0
        # Load icon, 90x90 pixel out of 120 pixel box
        imgSrc=f'units/{target}'
        template = cv2.imread(imgSrc)
        match=0
        # Compare each channel
        for i in range(3):
            img_channel=img_rgb[:,:,i]
            template_ch= template[:,:,i]
            # Compare images
            res = cv2.matchTemplate(img_channel, template_ch, cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            match += max_val/3
        if target == 'empty.png': # make empty slightly less likely
            match+=0.05
        current_icons.append([target,match])
    unit_df=pd.DataFrame(current_icons, columns=['icon_'+grid_id,'probability'])
    unit_df['probability']=unit_df['probability'].round(4)
    if not guess_unit:
        return unit_df
    if guess_unit:
        guess=unit_df.loc[unit_df['probability'].idxmin()]
        unit_pred = guess[0]
        if guess[1]>0.55:
            unit_pred = 'empty.png'
        return [unit_df.columns[0],unit_pred,guess[1]]


def polygon_test(file_name,i=4,unit=None):
    print(unit)
    gamma_offset_dict={
    'alchemist.png':150,
    'lightning.png':190,
    'hunter.png':185,
    'sharpshooter.png':140,
    'priest.png':160    
        }
    gamma_offset = gamma_offset_dict[unit] if unit in gamma_offset_dict else 180
    print(gamma_offset)
    # read image as grayscale
    img = cv2.imread(file_name,0)
    keep=35
    img[keep:-keep,keep:-keep] = 150
    img = cv2.GaussianBlur(img,(11,11),0)
    background= max(img[115:120].mean(),img[0:5].mean())
    #background= img[0:5].mean()
    thresh = cv2.threshold(img,gamma_offset,255,cv2.THRESH_BINARY)[1]
    #thresh = cv2.GaussianBlur(thresh,(11,11),0)
    if gamma_offset<150:
        thresh=255-thresh
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # Take largest polygon
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:4]
    i=3
    approx=cv2.approxPolyDP(cnts[0],1.5**i*0.01*cv2.arcLength(cnts[0],True),True)
    return approx,cnts,thresh,img


def get_poly(file_name,i=4,unit=None):
    gamma_offset_dict={
        'alchemist.png':10,
        'lightning.png':40,
        'hunter.png':15,
        'sharpshooter.png':-20,
        'priest.png':0    
            }
    gamma_offset = gamma_offset_dict[unit] if unit in gamma_offset_dict else 10
    # read image as grayscale
    img = cv2.imread(file_name,0)
    img = cv2.GaussianBlur(img,(11,11),0)
    background= max(img[115:120].mean(),img[0:5].mean())
    thresh = cv2.threshold(img,background+gamma_offset,255,cv2.THRESH_BINARY)[1]
    #thresh = cv2.GaussianBlur(thresh,(11,11),0)
    if gamma_offset<0:
        thresh=255-thresh
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # Take largest polygon
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    #Estimate corners
    i=3
    approx=cv2.approxPolyDP(cnts,1.5**i*0.01*cv2.arcLength(cnts,True),True)
    return approx


#truth=[2,4,1,2,3,1,2,5,3]
#result=np.array(guesses) != np.array(truth)
#correct = len(result)-sum(result)
#accuracy = correct*1000//len(result)/10
#
#print(f'Got {correct} / {len(result)} correct, {accuracy}% accuracy!')
#misses=result.nonzero()[0]
#if misses.size!=0:
#    display(curr_units_df[result])
#    figs=[show_contour(contour[polygon_id]) for polygon_id in misses]
#    display(Image.fromarray(np.hstack(figs)))



def spam_unit():
    bot.getScreen()
    for i in range(200):
        bot.getScreen()
        print("Step,", i)
        # get stored mana 
        mana=bot.getText(220,1360,90,50,new=False,digits=True)
        if mana=='': mana=0
        # get unit mana 
        mana_cost=bot.getText(450,1360,90,50,new=False,digits=True)
        print(mana,mana_cost)
        if mana_cost=='' and  mana==0:
            time.sleep(1)
            continue
        if int(mana)>int(mana_cost):
            print(f"Bought for {mana_cost}!")
            bot.click(450,1360)
            time.sleep(0.1)
        else: time.sleep(1)


def filter_keys(group_keys, token, exclude=False):
    if exclude:
        return [key for key in group_keys if not token in key]
    else:
        return [key for key in group_keys if token in key]

#tier_1s = filter_keys(group_keys,1)
#priest = filter_keys(group_keys,'priest.png')
#unit_list = set(tier_1s + priest)
#priest = unit_series.loc[['priest.png']]

def try_merge(prev_grid=None):
    info=''
    names=bot.scan_grid(new=True)
    grid_df=bot_perception.grid_status(names,prev_grid=prev_grid)
    df_split, df_groups, unit_list = bot_core.get_unit_count(grid_df)
    df_groups = df_groups.sort_values(ascending=False)
    # check if grid full
    if 'empty.png' in unit_list:
        # Merge if full board
        if df_groups['empty.png']<3:
            info='Merging!'
            unit_list.remove('empty.png')
            if 'hunter.png' in unit_list: unit_list.remove('hunter.png')
            for i in range(3):
                merge_tar = random.choice(unit_list)
                for j in range(3):
                    bot.merge_unit(grid_df,merge_tar)
                    time.sleep(0.2)
        else: info= 'need more units!'
    # If grid seems full, merge any
    else:
        info = 'Full Grid - Merging!'
        merge_tar = random.choice(unit_list)
        bot.merge_unit(grid_df,merge_tar)
    return grid_df,df_groups,info


def find_and_expand_unit_cluster(grid_df):
    from scipy import ndimage
    cluster_unit=grid_df['unit']=='engineer.png'
    binary=cluster_unit.astype(int).to_numpy()
    binary_2D= np.reshape(binary,(3,5))
    # Find largest cluster in grid
    loc = ndimage.find_objects(binary_2D)[0]
    # Create a matrix with 1 - 14
    pos_to_id=np.reshape(np.arange(15),(3,5))
    # Find location of all elements in cluster (square)
    cluster_id=pos_to_id[loc].flatten()
    # Select nonzero elements in square
    engineers_loc_cluster = np.flatnonzero(binary_2D[loc])
    # Get id of engineers
    engineers_id = cluster_id[engineers_loc_cluster]
    # Create blank grid and draw cluster
    blank_grid=np.zeros(15)
    blank_grid[engineers_id]=1
    cluster_grid = blank_grid.reshape(3,5)
    # Calculate adjacent squares in grid
    right_shift=np.pad(cluster_grid,((0,0),(1,0)), mode='constant')[:, :-1]
    left_shift =np.pad(cluster_grid,((0,0),(0,1)), mode='constant')[:, 1:]
    down_shift =np.pad(cluster_grid,((1,0),(0,0)), mode='constant')[:-1, :]
    up_shift =  np.pad(cluster_grid,((0,1),(0,0)), mode='constant')[1:, :]
    # Combine all shift and check all covered positions
    expanded_cluster=np.flatnonzero(up_shift + down_shift+left_shift+right_shift)
    new_loc=np.isin(expanded_cluster, engineers_id, invert=True)
    expanded_cluster = expanded_cluster[new_loc]
    # Draw cluster expansion locations
    expanded_grid=np.zeros(15)
    expanded_grid[expanded_cluster]=1
    expanded_grid = expanded_grid.reshape(3,5)
return expanded_grid

# get stored mana 
#bot.getText(220,1360,90,50,new=False,digits=True)

# get unit mana 
bot.getText(450,1360,90,50,new=False,digits=True)

# Get upgrade mana
#bot.getText(20,1540,700,50,new=False,digits=True).split('6')

# Remove tuple tokens from unit series
def remove_keys(unit_series,tokens = [('empty.png', 0)]):
    return unit_series[~unit_series.index.isin(tokens)]


# Try to find a merge target and merge it
def try_merge(self,rank=1,prev_grid=None):
    info=''
    merge_df =None
    names=self.scan_grid(new=True)
    grid_df=bot_perception.grid_status(names,prev_grid=prev_grid)
    df_split,unit_series, df_groups, group_keys=grid_meta_info(grid_df)
    # Select stuff to merge
    merge_series = unit_series[unit_series>=2] # At least 2 units ## ADD MIME to every count, use sample rank and check if mime exist
    # check if grid full
    if ('empty.png',0) in group_keys:
        # Try to merge priest
        merge_priest = filter_keys(merge_series,['priest.png'])
        if not merge_priest.empty:
            merge_df = self.merge_unit(df_split,merge_priest)
        # Merge if full board
        if df_groups['empty.png']<=2:
            info='Merging!'
            # Add criteria
            merge_series = filter_keys(merge_series,[rank,'priest.png'])
            if not merge_series.empty:
                # drop duplicated indices
                merge_series = merge_series[~merge_series.index.duplicated()]
                # Take index name of random sample
                merge_target =  merge_series.sample().index[0] 
                # Retrieve group    
                merge_df=df_split.get_group(merge_target)
                #merge_df=merge_df.sort_values(by='Age',ascending=False).reset_index(drop=True)
                # Send merge command
                merge_df = self.merge_unit(df_split,merge_series)
            else:
                info='not enough filtered targets!'
        else: info= 'need more units!'
    # If grid seems full, merge any
    else:
        info = 'Full Grid - Merging!'
        # Remove all high level crystals
        merge_df = self.merge_unit(df_split,merge_series)
    return grid_df,unit_series,merge_df,info

        if not len(series)==0:
            print(123)
            merge_series = pd.concat(series)
            if remove:
                merge_series = merge_series_temp[~merge_series_temp.index.isin(merge_series.index)]
                display(merge_series)
            else:
                continue
        elif not remove: # return empty list if empty and nothing matches criteria
            print(321)
            return pd.Series(dtype=object)
        # Otherwise Do next loop with unchanged merge series otherwise
    # Return result of all criterias
    print(555)
    return merge_series

# auto add training data



####
#### Legacy Unit rank recognition (used to bootstrap ML model)
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
    if len(cnts)==0:
        return None
    if num==1:
        cnts=cnts[0]
    return cnts

def get_poly(filename,ref,i=4,shape=(120, 120),debug=False):
    # read image as grayscale
    img = cv2.imread(filename,0)
    # Find edges in image
    edges = cv2.Canny(img,50,100)
    # read reference image
    img_ref = cv2.imread(ref,0)
    img_ref = cv2.GaussianBlur(img_ref,(5,5),0)
    # Apply filter
    img_match = edges &img_ref
    img_match = cv2.GaussianBlur(img_match,(5,5),0)
    # pick top polygon
    cnts=find_polygon(img_match,1)
    if cnts is None:
       return [[[  0,   0]]]
    # Approximate Polygon         # Change arclength to expect of rank polygon
    i=3
    approx=cv2.approxPolyDP(cnts,1.5**i*0.01*cv2.arcLength(cnts,True),True)
    if debug:
        return approx,cnts,img_match,edges,img
    if approx is None:
       return [[[  0,   0]]]
    return approx

# Find nearest coordinate in 2D using mean square error
def mean_square(array, coord):
    # take MSE (mean square error)
    coord_mse = ((array-coord)**2).mean(axis=2)
    idx=coord_mse.argmin()
    return coord_mse[idx][0]
# Run mean square of all ranks


# Returns guess of unit rank
def match_rank_old(filename):
    # Load dictionary with expected corner positions
    corner_dict={
    'rank2':np.array([[10,10],[110,110]]),
    'rank3':np.array([[5,10],[115,10],[60,110]]),
    'rank4':np.array([[60,0],[0,60],[60,120],[120,60]]),
    'rank5':np.array([[15,10],[15,80],[60,115],[105,80],[105,10]]),    
        }
    match_errors=[]
    for rank in corner_dict:
        target=f'unit_rank/{rank}_bin.png'
        # Get polygon of image
        target_corners=get_poly(filename,target)
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