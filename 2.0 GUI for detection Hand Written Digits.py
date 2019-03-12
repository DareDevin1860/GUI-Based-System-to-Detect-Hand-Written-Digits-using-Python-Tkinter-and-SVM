import tkinter as tk
from PIL import ImageTk,ImageDraw,Image

import PIL
import cv2
import numpy as np
from matplotlib import pyplot as plt

win=tk.Tk()

width=400
height=400
font1='Helvetica 10 bold'

from sklearn import datasets
digits=datasets.load_digits()
data=digits.data
target=digits.target
images=digits.images



from sklearn import svm

clsfr=svm.SVC(gamma=0.0001,C=100)
clsfr.fit(data,target)



def event_func(event):

    x1=event.x-25
    y1=event.y-25
    x2=event.x+25
    y2=event.y+25
    canvas1.create_oval((x1,y1,x2,y2),fill="black")
    img_draw.ellipse((x1,y1,x2,y2),fill="white")

def save():
    global img1
    img_array=np.array(img1)
    img_array=cv2.resize(img_array,(8,8))
    plt.imshow(img_array,cmap='binary')

    
    cv2.imwrite('digits.jpg',img_array)
    plt.show()
    

def clear():
    global img1,img_draw
    canvas1.delete('all')
    img1=PIL.Image.new('RGB',(width,height),(0,0,0))
    img_draw=ImageDraw.Draw(img1)

    

def predict():

    global img1,img_draw
    img_array=np.array(img1)
    img_array=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
    img_array=cv2.resize(img_array,(8,8))
    img_array_flat=img_array.ravel()#convert the 2d array into a 1D array

    img_array_flat=(img_array_flat/255.0)*15.0

    result=clsfr.predict([img_array_flat])
    label_predict.config(text='PREDICTED DIGIT : '+str(result))

    
    

canvas1=tk.Canvas(win,width=width,height=height,bg='white')
canvas1.grid(row=0,column=0,columnspan=4)

button_save=tk.Button(win,text='SAVE',bg='green',fg='white',font=font1,command=save)
button_save.grid(row=1,column=0,pady=5)

button_predict=tk.Button(win,text='PREDICT',bg='blue',fg='white',font=font1,command=predict)
button_predict.grid(row=1,column=1,pady=5)

button_clear=tk.Button(win,text='CLEAR',bg='grey20',fg='white',font=font1,command=clear)
button_clear.grid(row=1,column=2,pady=5)


button_exit=tk.Button(win,text='EXIT',bg='red',fg='white',font=font1,command=win.destroy)
button_exit.grid(row=1,column=3,pady=5)

label_predict=tk.Label(win,text='PREDICTED DIGIT: NONE',bg='gray90',font=font1)
label_predict.grid(row=2,column=0,columnspan=3)

canvas1.bind('<B1-Motion>',event_func)


img1=PIL.Image.new('RGB',(width,height),(0,0,0))
img_draw=ImageDraw.Draw(img1)
