from tkinter import *
import os
import numpy as np
import threading
import logging


def create_options(frame1,config):
    frame1.grid_rowconfigure(0, weight=1)
    frame1.grid_columnconfigure(0, weight=1)

    # General options
    label= Label(frame1,text="Options",justify=LEFT).grid(row=0, column=0, sticky=W)
    ads_var = IntVar() 
    pve_var = IntVar() 
    ad_check = Checkbutton(frame1, text='Watch ads', variable=ads_var,justify=LEFT).grid(row=1, column=0, sticky=W)
    pve_check = Checkbutton(frame1, text='PvE', variable=pve_var,justify=LEFT).grid(row=1, column=1, sticky=W)
    # Mana level targets
    mana_label= Label(frame1,text="Mana Level Targets",justify=LEFT).grid(row = 2, column = 0, sticky=W)
    stored_values=np.fromstring(config['bot']['mana_level'], dtype=int, sep=',') 
    mana_vars = [IntVar(value=int(i in stored_values)) for i in range(1,6)]
    mana_buttons = [Checkbutton(frame1, text=f'Card {i+1}', variable=mana_vars[i],justify=LEFT).grid(row = 2, column = i+1) for i in range(5)]
    # Dungeon Floor
    floor_label= Label(frame1,text="Dungeon Floor",justify=LEFT).grid(row = 3, column = 0, sticky=W)
    floor = Entry(frame1, name='floor_entry', width=5)
    if config.has_option('bot', 'floor'):
        floor.insert(0,config['bot']['floor'])
    floor.grid(row = 3, column = 1)
    return ads_var, pve_var, mana_vars, floor

def create_combat_info(frame2):
    # Create text widgets
    grid_dump = Text(frame2, height = 18, width = 60,bg='#575559',fg='#ffffff')
    unit_dump = Text(frame2, height = 10, width = 30,bg='#575559',fg='#ffffff')
    merge_dump = Text(frame2, height = 10, width = 30,bg='#575559',fg='#ffffff')
    grid_dump.grid(row=0, sticky=S)
    unit_dump.grid(row=1, column=0, sticky=W)
    merge_dump.grid(row=1, column=0, sticky=E)
    return grid_dump, unit_dump, merge_dump

def create_base():
    root=Tk()
    root.title("RR bot")
    root.geometry("800x600")
    # Set dark background
    root.configure(background='#575559')
    # Set window icon to png
    root.iconbitmap('calculon.ico')
    root.resizable(False, False) # ai
    # Add frames
    frame1 = Frame(root)
    frame2 = Frame(root)
    frame2.grid_rowconfigure(0, weight=1)
    frame2.grid_columnconfigure(0, weight=1)
    frame3 = Frame(root,bg='#575559')
    frame3.grid_columnconfigure(0, weight=1)
    frame4 = Frame(root)
    return root

def update_text(i,combat,output,grid_df,unit_series,merge_series,df_groups,info):
    # info + general info
    grid_df['unit'] = grid_df['unit'].apply(lambda x: x.replace('.png',''))
    write_to_widget(grid_dump, f"{combat}, {i+1}/8 {output[1]}, {info} Average age: {str(grid_df['Age'].mean().round(2))}'\n{grid_df.to_string()}\n{'#'*60}")
    if unit_series is not None:
        write_to_widget(unit_dump, unit_series.to_string())
    if merge_series is not None:
        write_to_widget(merge_dump, merge_series.to_string())

# Function to update text widgets
def write_to_widget(root, tbox, text):
    tbox.config(state=NORMAL)
    tbox.delete(1.0,END)
    tbox.insert(END, text)
    tbox.config(state=DISABLED)
    root.update_idletasks()



class TextHandler(logging.StreamHandler):
    def __init__(self, textctrl):
        logging.StreamHandler.__init__(self) # initialize parent
        self.textctrl = textctrl

    def emit(self, record):
        msg = self.format(record)
        self.textctrl.config(state="normal")
        self.textctrl.insert("end", msg + "\n")
        self.flush()
        # scroll to the bottom
        self.textctrl.see("end")
        self.textctrl.config(state="disabled")

