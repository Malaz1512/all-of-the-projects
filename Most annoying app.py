from tkinter import *

wn1 = Tk()

def create_screen():
    wn2 = Tk()
    def Buttons2():
        button2 = Button(wn2, text="Close Screen")
        button2.config(command=create_screen)
        button2.pack()
    Buttons2()
    wn2.mainloop()

def Buttons():
    button = Button(wn1, text="Close Screen")
    button.config(command=create_screen)
    button.pack()

Buttons()

while True:
    if wn1.quit:
        wn1 = Tk()
        def Buttons():
            button = Button(wn1, text="Close Screen")
            button.config(command=create_screen)
            button.pack()
        Buttons()
        wn1.mainloop()

    wn1.mainloop()
