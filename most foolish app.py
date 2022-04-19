# Malaz OS
# Version: 0.0.1
# languages used: python
# authors/creators: Malaz Nakaweh

import turtle
from ursina import *
import time
import random

wn = turtle.Screen()
wn.bgcolor("blue")
wn.setup(width=1920, height=1080)
wn.title("Malaz OS Setup")
wn.tracer(0)

welcome = turtle.Turtle()
welcome.hideturtle()
welcome.color("black")
welcome.penup()
welcome.goto(0, 200)
welcome.write("Welcome to the setup, press start to download", align="center", font=("Courier", 24, "normal"))

pen = turtle.Turtle()
pen.hideturtle()
pen.pencolor('#111111')
pen.fillcolor('grey')

Button_x = -48
Button_y = -35
ButtonLength = 100
ButtonWidth = 50

def draw_rect_button(pen, message = '   Start'):
    pen.penup()
    pen.begin_fill()
    pen.goto(Button_x, Button_y)
    pen.goto(Button_x + ButtonLength, Button_y)
    pen.goto(Button_x + ButtonLength, Button_y + ButtonWidth)
    pen.goto(Button_x, Button_y + ButtonWidth)
    pen.goto(Button_x, Button_y)
    pen.end_fill()
    pen.goto(Button_x, Button_y + 10)
    pen.write(message, font = ('Arial', 15, 'normal'))

draw_rect_button(pen)

def buttonClick(x, y):
    if Button_x <= x <= Button_x + ButtonLength:
        if Button_y <= y <= Button_y + ButtonWidth:
            pen.clear()
            welcome.clear()
            num = random.randint(1, 2)
            pen2 = turtle.Turtle()
            pen2.hideturtle()
            pen2.penup()
            pen2.color("black")
            pen2.goto(0,0)
            pen2.write("Downloading Malaz OS", align="center", font=('Courier', 24, 'normal'))
            time.sleep(num)
            pen2.clear()
            num = random.randint(1, 2)
            pen2.write("Downloading Malaz OS.", align="center", font=('Courier', 24, 'normal'))
            time.sleep(num)
            pen2.clear()
            num = random.randint(1, 2)
            pen2.write("Downloading Malaz OS..", align="center", font=('Courier', 24, 'normal'))
            time.sleep(num)
            pen2.clear()
            num = random.randint(1, 2)
            pen2.write("Downloading Malaz OS...", align="center", font=('Courier', 24, 'normal'))
            time.sleep(num)
            pen2.clear()
            pen2.goto(0, 200)
            pen2.write("Task Failed Successfully, please restart your computer", align="center", font=('Courier', 24, 'normal'))

while True:
    wn.onscreenclick(buttonClick)
    wn.listen()

    wn.mainloop()