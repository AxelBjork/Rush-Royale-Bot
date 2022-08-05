from tkinter import *
import os
import numpy as np
import threading
import logging
import configparser

# internal
import bot_handler



# GUI Class
class RR_bot:
    def __init__(self):
        # State variables
        self.stop_flag = False
        self.running = False
        self.bot_initalized = threading.Event()
        self.info_ready = threading.Event()
        # Read config file
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        # Setup logger
        logging.basicConfig(filename='RR_bot.log',level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.handlers.clear()
        # Create tkinter window base
        self.root = create_base()
        self.frames = self.root.winfo_children()
        # Setup frame 1 (options)
        self.ads_var, self.pve_var, self.mana_vars, self.floor = create_options(self.frames[0],self.config)
        # Setup frame 2 (combat info)
        self.grid_dump, self.unit_dump, self.merge_dump = create_combat_info(self.frames[1])
        ## rest need to be cleaned up
        # Log frame
        logger_feed = Text(self.frames[3], height = 30, width = 38,bg='#575559',fg='#ffffff',wrap=WORD, font=('Consolas',9))
        logger_feed.grid(row=0, sticky=S)
        # Connect logger to text widget
        guiHandler = TextHandler(logger_feed)
        self.formatter = logging.Formatter('[%(asctime)s] %(message)s',"%H:%M")
        guiHandler.setFormatter(self.formatter)
        self.logger.addHandler(guiHandler)
        start_button = Button(self.frames[2],text="Start Bot",command=self.start_command)
        stop_button = Button(self.frames[2], text='Stop Bot', command=self.stop_bot,padx=20)
        start_button.grid(row=0, column=1, padx= 10)
        stop_button.grid(row=0, column=2, padx= 5)

        self.frames[0].pack(padx=0,pady=0,side=TOP, anchor=NW)
        self.frames[1].pack(padx=10,pady=10,side=RIGHT, anchor=SE)
        self.frames[2].pack(padx=10,pady=10,side=BOTTOM, anchor=SW)
        self.frames[3].pack(padx=10,pady=10,side=LEFT, anchor=SW)
        self.root.mainloop()
    
    # Clear loggers, collect threads, and close window
    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.info('Exiting GUI')
        self.logger.handlers.clear()
        self.thread_run.join()
        self.thread_init.join()
        self.root.destroy()
    # Initilzie the thread for main bot
    def start_command(self):
        self.logger.info(f"Logging with logger: {self.logger.name}")
        self.stop_flag = False
        floor_var=int(self.floor.get())
        card_level = [var.get() for var in self.mana_vars]*np.arange(1,6)
        card_level = card_level[card_level != 0]
        self.config['bot']['floor'] = str(floor_var)
        self.config['bot']['mana_level'] = np.array2string(card_level,separator=',')[1:-1]
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        # Dump config
        self.logger.info("Stored settings to config!")
        if self.running:
            return
        self.running = True
        # Start initalization thread if not already done
        if not self.bot_initalized.isSet():
            self.thread_init = threading.Thread(target=self.init_bot, args=())
            self.thread_init.start()
        # Start main thread which will wait on bot_initalized
        self.thread_run = threading.Thread(target=self.start_bot, args=())
        self.thread_run.start()
    # Initialize the bot
    def init_bot(self):
        self.logger.info('Starting bot...')
        self.bot_instance = bot_handler.start_bot_class(self.logger)
        self.bot_initalized.set()
    # Run the bot
    def start_bot(self):
        # Wait for started thread to be done
        self.bot_initalized.wait(timeout=20)
        self.thread_init.join() # Eat finished thread
        self.logger.info('Started bot!')
        os.system('cls')
        os.system("type src\startup_message.txt")
        infos_ready = threading.Event()
        self.bot_instance.bot_stop = False
        self.bot_instance.logger = self.logger
        bot = self.bot_instance
        # Start bot thread
        thread_bot = threading.Thread(target=bot_handler.bot_loop, args=([bot,infos_ready]))
        thread_bot.start()
        # Dump infos to gui whenever ready
        while(1):
            infos_ready.wait(timeout=5)
            self.update_text(bot.combat_step,bot.combat,bot.output,bot.grid_df,bot.unit_series,bot.merge_series,bot.df_groups,bot.info)
            infos_ready.clear()
            if self.stop_flag:
                self.bot_instance.bot_stop = True
                self.logger.info('Exiting main loop...')
                thread_bot.join()
                self.logger.info('Bot stopped!')
                self.logger.info('Safe to close gui')
                return
    # Raise stop flag to threads
    def stop_bot(self):
        self.running = False
        self.stop_flag = True
        self.logger.info('Stopping bot!')
        
    # Update text widgets with latest info
    def update_text(self, i,combat,output,grid_df,unit_series,merge_series,df_groups,info):
        # info + general info
        if grid_df is not None:
            grid_df['unit'] = grid_df['unit'].apply(lambda x: x.replace('.png',''))
            grid_df['unit'] = grid_df['unit'].apply(lambda x: x.replace('empty','-'))
            write_to_widget(self.root, self.grid_dump, f"{combat}, {i+1}/8 {output}, {info}\n{grid_df.to_string()}\nAverage age: {str(grid_df['Age'].mean().round(2))}")
        if unit_series is not None:
            #unit_series['unit'] = unit_series['unit'].apply(lambda x: x.replace('.png',''))
            write_to_widget(self.root,self.unit_dump, unit_series.to_string())
        if merge_series is not None:
            #merge_series['unit'] = merge_series['unit'].apply(lambda x: x.replace('.png',''))
            write_to_widget(self.root,self.merge_dump, merge_series.to_string())
        if df_groups is not None:
            df_groups['unit'] = df_groups['unit'].apply(lambda x: x.replace('.png',''))
            unit1, unit2 = df_groups.iloc[0:2]['unit']
            rank  = df_groups.iloc[0]['rank']
            self.logger.info(f"Rank {rank} {unit1}-> {unit2}")

###
### END OF GUI CLASS
###

def create_options(frame1,config):
    frame1.grid_rowconfigure(0, weight=1)
    frame1.grid_columnconfigure(0, weight=1)

    # General options
    label= Label(frame1,text="Options",justify=LEFT).grid(row=0, column=0, sticky=W)
    ads_var = IntVar() 
    pve_var = IntVar() 
    #ad_check = Checkbutton(frame1, text='Watch ads', variable=ads_var,justify=LEFT).grid(row=1, column=0, sticky=W)
    #pve_check = Checkbutton(frame1, text='PvE', variable=pve_var,justify=LEFT).grid(row=1, column=1, sticky=W)
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

# Start the actual bot
bot_gui = RR_bot()