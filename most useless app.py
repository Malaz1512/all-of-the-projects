from tkinter import *

wn = Tk()

def ButtonClick():
    button = Button(wn, text="oh hi")
    button.config(command=Buttons)
    button.pack()

def Buttons():
    button = Button(wn, text="hi")
    button.config(command=ButtonClick)
    button.pack()

Buttons()
 
wn.mainloop()