import tkinter as tk
from View.MainWindow import Window
import threading



'''
Main module to run GUI
'''

def run():
    root=tk.Tk()
    root.title("Final Project")
    root.geometry("1200x650")
    root.resizable(0, 0)
    window=Window(root)



    root.mainloop()

if __name__ == '__main__':
    main_thread = threading.Thread(target=run)
    main_thread.start()