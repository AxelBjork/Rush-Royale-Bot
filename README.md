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


![high_board](https://user-images.githubusercontent.com/71280183/178340847-3c03ccb3-840c-4a4d-ba89-e2ac3d5883b7.png)


## Setup Guide

**Python**

Install Latest Python 3.9 (Windows installer 64-bit)

https://www.python.org/downloads/

Select add Python to path, check `python --version`  works and gives Python 3.9.13

Download and extract this repo

**Bluestacks**

Install Latest Bluestacks 5

Settings:

(Graphics) Graphics engine mode: Compatibility

(Graphics) Graphics renderer: OpenGL

(Graphics) Interface renderer: Auto

(Advanced) Android Debug Bridge: Enabled - Note the port number here

Setup google account, download rush royale, ect.

**Bot**

run install.bat to create repo and install dependencies

run lanch_gui.bat

(temp) units and other settings have to be configured in bot_handler.py, this will be moved to the config.ini file.
