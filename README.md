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

Install requirements 

`pip install -r requirements.txt`

**Bluestacks**

Install Latest Bluestacks 5

Settings:

(Graphics) Graphics engine mode: Compatibility

(Graphics) Graphics renderer: OpenGL

(Graphics) Interface renderer: Auto

(Advanced) Android Debug Bridge: Enabled - Note the port number here

Setup google acconut, download rush royale, ect.

**Scrcpy**

Download Scrcpy from https://github.com/Genymobile/scrcpy

Extract to suitable directory

Open command prompt and verify scrcpy + Bluestacks works

`directory\scrcpy -s 127.0.0.1:xxxxx`

`directory\scrcpy -s emulator-5554` (if the port was 5555)

Add the directory to bot config.ini

**Launch Bot**

Open terminal and navigate to bot directory and launch notebook

`cd C:\Code\Rush-Royale-Bot`

`Jupyter notebook`

In the new browser open RR_bot.ipynb

Change `sel_units` to your units

Run first cell (SETUP). Scrcpy can be closed once opened

Change list in `bot.mana_level([2,3,5],hero_power=True)` to the card slots you want the bot to mana level

Change `merge_target` to main dps unit which the bot will try to perserve and rank up with dryad.

Change floor in `bot.battle_screen(start=True,pve=True,floor=7) #(only 1,2,4,5,7,8,10 possible)` to floor bot should farm.

Run second cell (RUN BOT)
