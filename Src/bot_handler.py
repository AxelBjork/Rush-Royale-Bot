import os
import time
import logging
from subprocess import Popen,DEVNULL
# Image processing
import cv2
# internal
import port_scan
import bot_core
import bot_perception

import zipfile
import functools
import pathlib
import shutil
import requests
from tqdm.auto import tqdm
# from here https://stackoverflow.com/a/63831344
def download(url, filename):
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get('Content-Length', 0))

    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)

    return path

# Moves selected units from collection folder to deck folder for unit recognition options
def select_units(units,show=False):
    if show:
        print(os.listdir("all_units")) 
        print('Chosen:\n',units)
    if os.path.isdir('units'):
        [os.remove('units/'+unit) for unit in os.listdir("units")]
    else: os.mkdir('units')
    # Read and write all images
    for new_unit in units:
        cv2.imwrite('units/'+new_unit,cv2.imread('all_units/'+new_unit))


def start_bot_class(logger):
    # auto-install scrcpy if needed
    if not check_scrcpy(logger):
        return None
    device = port_scan.get_device()
    if device is None:
        raise Exception("No device found!")
    # Start Scrcpy once per restart
    if 'started_scrcpy' not in globals():
        global started_scrcpy
        started_scrcpy=True
        logger.info(f'Connecting Scrcpy to {device}')
        proc = Popen(['.scrcpy/scrcpy','-s',device],stdout=DEVNULL)
        time.sleep(1) # <-- sleep for 1 second
        proc.terminate() # <-- terminate the process (Scrcpy window i2s not needed)
    sel_units= ['chemist.png','knight_statue.png','harlequin.png','dryad.png','demon_hunter.png']
    select_units(sel_units,show=False)
    bot = bot_core.Bot(device)
    return bot

# Loop for combat actions
def combat_loop(bot,grid_df,mana_targets,merge_target='demon_hunter.png'):
    time.sleep(0.2)
    # Upgrade units
    bot.mana_level([2,3,5],hero_power=True)
    # Spawn units
    bot.click(450,1360)
    # Try to merge units
    grid_df,unit_series,merge_series,df_groups,info = bot.try_merge(prev_grid=grid_df,merge_target='demon_hunter.png')
    return grid_df,unit_series,merge_series,df_groups,info

# Run the bot
def bot_loop(bot,info_event):
    wait=0
    combat = 0
    grid_df =None
    watch_ad = False
    train_ai = False
    # Main loop
    bot.logger.info(f'Bot mainloop started')
    while(not bot.bot_stop):
        output = bot.battle_screen(start=False)
        if output[1]=='fighting':
            watch_ad = True 
            wait = 0
            combat+=1 
            if combat>50:
                bot.restart_RR()
                combat = 0
                continue
            for i in range(8):
                grid_df,bot.unit_series,bot.merge_series,bot.df_groups,bot.info = combat_loop(bot,grid_df,mana_targets = [2,3,5],merge_target='demon_hunter.png')
                bot.grid_df = grid_df.copy()
                bot.combat = combat
                bot.output = output[1]
                bot.combat_step=i
                info_event.set()
                if bot.bot_stop:
                    return
            # Wait until late stage in combat and if consistency is ok, not stagnate save all units for ML model
            if combat==25 and 5<grid_df['Age'].mean()<50 and train_ai:
                bot_perception.add_grid_to_dataset()
        elif output[1]=='home' and watch_ad:
            for i in range(3):
                bot.watch_ads()
            watch_ad = False
        else:
            combat=0
            output = bot.battle_screen(start=True,pve=True,floor=7) #(only 1,2,4,5,7,8,10 possible)
            wait+=1
            if wait>40:
                bot.logger.info('RESTARTING')
                bot.restart_RR(),
                wait=0
            bot.logger.info(f'{output[1]}, wait count: {wait}')

def check_scrcpy(logger):
    if os.path.exists('.scrcpy/scrcpy.exe'):
        logger.info('scrcpy is installed')
        return True
    else:
        logger.info('scrcpy is not installed')
        # Download 
        download('https://github.com/Genymobile/scrcpy/releases/download/v1.24/scrcpy-win64-v1.24.zip', 'scrcpy.zip')
        with zipfile.ZipFile('scrcpy.zip', 'r') as zip_ref:
            zip_ref.extractall('.scrcpy')
        # Verify 
        if os.path.exists('.scrcpy/scrcpy.exe'):
            logger.info('scrcpy succesfully installed')
            # remove zip file
            os.remove('scrcpy.zip')
            return True