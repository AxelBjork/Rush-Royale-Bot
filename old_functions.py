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