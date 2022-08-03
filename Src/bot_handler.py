import os
import time
import logging
from subprocess import Popen
# Image processing
import cv2
# internal
import port_scan

# Move selected units from collection folder to deck folder for unit recognition options
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

def setup_logger():
    logging.basicConfig(filename='RR_bot.log',level=logging.INFO)
    # Delete previous log file
    if os.path.exists('RR_bot.log'):
        try:
            os.remove('RR_bot.log')
        except PermissionError:
            print('Log file is already open')
            return
    # Set log format and dateformat
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__+'.RR_Bot')
    logger.info('Initializing bot')
    return logger

def start_bot_class(config):
    scrcpy_path=config['bot']['scrcpy_path']
    device = port_scan.get_device()
    if device is None:
        raise Exception("No device found!")
    # Start Scrcpy once per restart
    if 'started_scrcpy' not in globals():
        global started_scrcpy
        started_scrcpy=True
        proc = Popen([os.path.join(scrcpy_path,'scrcpy'),'-s',device], shell=True)
        time.sleep(1) # <-- sleep for 1 second
        proc.terminate() # <-- terminate the process (Scrcpy window can be closed)
    sel_units= ['chemist.png','knight_statue.png','harlequin.png','dryad.png','demon_hunter.png']
    select_units(sel_units,show=False)
    bot = Bot(device)
    return bot