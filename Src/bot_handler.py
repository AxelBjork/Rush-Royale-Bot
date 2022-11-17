import os
import time
import numpy as np
import logging
from subprocess import Popen, DEVNULL
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
def select_units(units):
    if os.path.isdir('units'):
        [os.remove('units/' + unit) for unit in os.listdir("units")]
    else:
        os.mkdir('units')
    # Read and write all images
    for new_unit in units:
        try:
            cv2.imwrite('units/' + new_unit, cv2.imread('all_units/' + new_unit))
        except Exception as e:
            print(e)
            print(f'{new_unit} not found')
            continue
    # Verify enough units were selected
    return len(os.listdir("units")) > 4


def start_bot_class(logger):
    # auto-install scrcpy if needed
    if not check_scrcpy(logger):
        return None
    bot = bot_core.Bot()
    return bot


# Loop for combat actions
def combat_loop(bot, grid_df, mana_targets, user_target='demon_hunter.png'):
    time.sleep(0.2)
    # Upgrade units
    bot.mana_level(mana_targets, hero_power=True)
    # Spawn units
    bot.click(450, 1360)
    # Try to merge units
    grid_df, unit_series, merge_series, df_groups, info = bot.try_merge(prev_grid=grid_df, merge_target=user_target)
    return grid_df, unit_series, merge_series, df_groups, info


# Run the bot
def bot_loop(bot, info_event):
    # Load user config
    config = bot.config['bot']
    user_pve = config.getboolean('pve', True)
    bot.logger.warning(f'PVE is set to {user_pve}')
    user_floor = int(config.get('floor', 5))
    user_level = np.fromstring(config['mana_level'], dtype=int, sep=',')
    user_target = config['dps_unit'].split('.')[0] + '.png'
    # Load optional settings
    require_shaman = config.getboolean('require_shaman', False)
    max_loops = int(config.get('max_loops', 800))  # this will increase time waiting when logging in from mobile
    # Dev options (only adds images to dataset, rank ai can be trained with bot_perception.quick_train_model)
    train_ai = False
    # State variables
    wait = 0
    combat = 0
    watch_ad = False
    grid_df = None
    # Wait for login
    time.sleep(5)
    # Main loop
    bot.logger.debug(f'Bot mainloop started')
    # Wait for game to load
    while (not bot.bot_stop):
        # Fetch screen and check state
        output = bot.battle_screen(start=False)
        if output[1] == 'fighting':
            watch_ad = True
            wait = 0
            combat += 1
            if combat > max_loops:
                bot.restart_RR()
                combat = 0
                continue
            elif bot.bot_stop:
                return
            elif require_shaman and not (output[0] == 'shaman_opponent.png').any(axis=None):
                bot.logger.info('Shaman not found, checking again...')
                if any([(bot.battle_screen(start=False)[0] == 'shaman_opponent.png').any(axis=None) for i in range(1)]):
                    continue
                bot.logger.warning('Leaving game')
                bot.restart_RR(quick_disconnect=True)
            # Combat Section
            grid_df, bot.unit_series, bot.merge_series, bot.df_groups, bot.info = combat_loop(
                bot, grid_df, user_level, user_target)
            bot.grid_df = grid_df.copy()
            bot.combat = combat
            bot.output = output[1]
            bot.combat_step = 1
            info_event.set()
            # Wait until late stage in combat and if consistency is ok, not stagnate save all units for ML model
            if combat == 25 and 5 < grid_df['Age'].mean() < 50 and train_ai:
                bot_perception.add_grid_to_dataset()
        elif output[1] == 'home' and watch_ad:
            [bot.watch_ads() for i in range(3)]
            watch_ad = False
        else:
            combat = 0
            bot.logger.info(f'{output[1]}, wait count: {wait}')
            output = bot.battle_screen(start=True, pve=user_pve, floor=user_floor)
            wait += 1
            if wait > 40:
                bot.logger.info('RESTARTING')
                bot.restart_RR(),
                wait = 0


def check_scrcpy(logger):
    if os.path.exists('.scrcpy/scrcpy.exe'):
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