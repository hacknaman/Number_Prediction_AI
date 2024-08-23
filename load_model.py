from tkinter import *
from tkinter import colorchooser
import PIL.ImageGrab as ImageGrab
from tkinter import filedialog
from tkinter import messagebox

import tensorflow as tf

import numpy as np

root = Tk()
root.title("Predict App")
root.geometry("400x470")

# -------------- variables --------------------

stroke_size = IntVar()
stroke_size.set(30)

stroke_color = StringVar()
stroke_color.set("white")

# variables for pencil 
prevPoint = [0,0]
currentPoint = [0,0] 

# variable for text
textValue = StringVar()

model = tf.keras.models.load_model('my_model.keras')

def usePredict():
    x = root.winfo_rootx()
    y = root.winfo_rooty()+70
    img = ImageGrab.grab(bbox=(x,y,x+400,y+400)).convert("L")
    img = img.resize((20, 20))

    img_np = np.array(img).astype(np.uint8)
    img_np[img_np < 20] = 0

    np.set_printoptions(precision=2, suppress=True, linewidth=150)
    
    print ('The shape of element of X is: ' + str(img_np.shape))
    print ('The first element of X is: \n', img_np )
    
    # Predict using the Neural Network
    prediction = model.predict(img_np.reshape(1,400))
    prediction_p = tf.nn.softmax(prediction)
    yhat = np.argmax(prediction_p)
    print(f'pridect num in drawing is {yhat}')
    messagebox.showinfo("Predict App" , f"Number is {yhat}.")

def useClear():
    canvas.delete('all')
    print(f'clear drawing area')

def paint(event):
    global prevPoint
    global currentPoint
    x = event.x
    y = event.y
    currentPoint = [x,y]

    if prevPoint != [0,0] : 
        canvas.create_polygon(prevPoint[0] , prevPoint[1] , currentPoint[0] , currentPoint[1],fill=stroke_color.get() , outline=stroke_color.get() , width=stroke_size.get())        

    prevPoint = currentPoint

    if event.type == "5" :
        prevPoint = [0,0]


def saveImage():
    try:
        x = root.winfo_rootx()
        y = root.winfo_rooty()+70
        img = ImageGrab.grab(bbox=(x,y,x+400,y+400)).convert("L")
        img = img.resize((20, 20))
        img.save(f'PngData/Saved.png')
        showImage = messagebox.askyesno("Predict App" , "Do you want to open image?")
        if showImage:
            img.show()

    except Exception as e:
        messagebox.showinfo("Predict app: " , "Error occured")


def clear():
    if messagebox.askokcancel("Predict app" , "Do you want to clear everything?"):
        canvas.delete('all')

def help():
    helpText = "1. Draw by holding right button of mouse to draw number. \n2.Click Predict to predict the number what is drawn \n3. Click on Clear to clear entire Canvas\n4. Click on Save to save the image in PngData folder"
    messagebox.showinfo("Help" , helpText)


def about():
    messagebox.showinfo("About" , "This Predict app is developed by Naman!")

def writeText(event):
    canvas.create_text(event.x , event.y , text=textValue.get())
# ------------------- User Interface -------------------

# Frame - 1 : Tools 

frame1 = Frame(root , height=400 , width=400 )
frame1.grid(row=0 , column=0, sticky=NW)

# toolsFrame 

toolsFrame = Frame(frame1 , height=100 , width=100, relief=SUNKEN , borderwidth=2)
toolsFrame.grid(row=0 , column=0 )
toolsLabel = Label(toolsFrame , text="Action", width=10)
toolsLabel.grid(row=0 , column=0)
pencilButton = Button(toolsFrame , text="Predict" , width=10 , command=usePredict)
pencilButton.grid(row=0 , column=1)
eraserButton = Button(toolsFrame , text="Clear" , width=10 , command=useClear)
eraserButton.grid(row=0 , column=2)


# helpSettingFrame

helpSettingFrame = Frame(frame1, height=100 , width=100, relief=SUNKEN , borderwidth=2)
helpSettingFrame.grid(row = 1 , column=0)
saveImageButton = Button(helpSettingFrame , text="Save" , bg="white" , width=10 , command=saveImage)
saveImageButton.grid(row=0 , column=0)
helpButton = Button(helpSettingFrame , text="Help" , bg="white" , width=10 , command=help)
helpButton.grid(row=0 , column=1)
aboutButton = Button(helpSettingFrame , text="About" , bg="white" , width=10 , command=about)
aboutButton.grid(row=0 , column=2)

# Frame - 2 - Canvas

frame2 = Frame(root , height=400 , width=400 , bg="yellow")
frame2.grid(row=1 , column=0)

canvas = Canvas(frame2 , height=400 , width=400 , bg="black" )
canvas.grid(row=0 , column=0)
canvas.bind("<B1-Motion>", paint)
canvas.bind("<ButtonRelease-1>", paint)

root.resizable(False , False)
root.mainloop()