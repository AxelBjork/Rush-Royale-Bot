# Rush-Royale-Bot
Python based bot for Rush Royale

Use with Bluestacks on PC

## Farm unlimited gold!
* Can run 24/7 and allow you to easily upgrade all availble units with gold to spare.
* Optimized to farm dungeon floor 5 

## Functionality 
* Can send low latency commands to game via Scrpy ADB
* Jupyter notebook for interacting, adding new units
* Automatically refreshes store, watches ads, completes quests, collects ad chest
* Unit type detection with openCV: ORB detector
* Rank detection with sklearn LogisticRegression (Very accurate)

![output](https://user-images.githubusercontent.com/71280183/171181226-d680e7ca-729f-4c3d-8fc6-573736371dfb.png)

![new_gui](https://user-images.githubusercontent.com/71280183/183141310-841b100a-2ddb-4f59-a6d9-4c7789ba72db.png)



## Setup Guide

**Python**

Install Latest Python 3.9 (Windows installer 64-bit)

https://www.python.org/downloads/ (windows 64-bit installer)[https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe]

Select add Python to path, check `python --version`  works and gives Python 3.9.13

Download and extract this repo

**Bluestacks**

Install Latest Bluestacks 5

Settings:

(Display) Resolution: 1600 x 900

(Graphics) Graphics engine mode: Compatibility (this can help if you have issues with scrcpy)

(Advanced) Android Debug Bridge: Enabled - Note the port number here

Setup google account, download rush royale, ect.

**Bot**

run install.bat to create repo and install dependencies

run lanch_gui.bat

(temp) units and other settings have to be configured in bot_handler.py, this will be moved to the config.ini file.
