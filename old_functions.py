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