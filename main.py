
import neural_network_class as nc
import numpy as np
import copy
import random
from tkinter import *


net = nc.NeuralNetwork([784, 30, 10])
for i in range(0, len(net.weights)):
    net.weights[i] = np.loadtxt('weights' + str(i) + '.csv', delimiter=',')


##TEST##
#test_data = input.load_data_wrapper()[2]
#print(net.evaluate(test_data))


##GUI##
input_matrix = np.zeros((784, 1))
input_hd = np.zeros((84,84))

def colorIn(event):
    x = event.x
    y = event.y
    index_x = int((x-40)/5)
    index_y = int((y - 40) / 5)
    global input_hd
    input_hd[index_y][index_x] = 1
    for i in range(-2, 3):
        for j in range(-2, 3):
            input_hd[index_y+i][index_x+j] = 1
    for i in range(-2, 3):
        for j in range(-2, 3):
            w.create_rectangle(40+((index_x+j)*5), 40+((index_y+i)*5), 40+((index_x+j)*5)+5, 40+((index_y+i)*5)+5, width=0, fill="black")

def erase(event):
    global input_matrix
    global input_hd
    input_matrix = np.zeros((784, 1))
    input_hd = np.zeros((84, 84))
    for i in range(0, 84):
        for j in range(0, 84):
            w.create_rectangle(40 + 5 * j, 40 + 5 * i, 45 + 5 * j, 45 + 5 * i, width=0, fill="white")
    for i in range(0, 12):
        for j in range(0, 84):
            w.create_rectangle(40 + 5 * j, 40 + 5 * i, 45 + 5 * j, 45 + 5 * i, width=0, fill="gray88")
    for i in range(72, 84):
        for j in range(0, 84):
            w.create_rectangle(40 + 5 * j, 40 + 5 * i, 45 + 5 * j, 45 + 5 * i, width=0, fill="gray88")
    for i in range(12, 72):
        for j in range(0, 12):
            w.create_rectangle(40 + 5 * j, 40 + 5 * i, 45 + 5 * j, 45 + 5 * i, width=0, fill="gray88")
    for i in range(12, 72):
        for j in range(72, 84):
            w.create_rectangle(40 + 5 * j, 40 + 5 * i, 45 + 5 * j, 45 + 5 * i, width=0, fill="gray88")
    w.create_rectangle(220, 510, 280, 570, width=0, fill="white")

def read(event):
    global input_matrix
    global input_hd
    for i in range(0, 28):
        for j in range(0,28):
            input_matrix[j + 28*i][0] = (1/9) * (input_hd[3*i][3*j] + input_hd[3*i][3*j+1] + input_hd[3*i][3*j+2] + input_hd[3*i+1][3*j] + input_hd[3*i+1][3*j+1] + input_hd[3*i+1][3*j+2] + input_hd[3*i+2][3*j] + input_hd[3*i+2][3*j+1] + input_hd[3*i+2][3*j+2] )
    k = 0
    for i in range(0, 28):
        for j in range(0, 28):
            w.create_rectangle(40 + 15 * j, 40 + 15 * i, 55 + 15 * j, 55 + 15 * i, width=0,
                               fill="grey" + str(int(99 - (input_matrix[k][0]) * 98)))
            k += 1
    net.feedforward(input_matrix)
    #print(np.argmax(net.a[-1]))
    w.create_text(250, 540, fill="grey20", font="Arial 40", text=str(np.argmax(net.a[-1])))

root = Tk()     #create tkinter object, blank window
root.title("")
w = Canvas(root, width=500, height=620)
w.configure(bg='grey20')
w.grid(row=0, column=0)
for i in range(0, 84):
    for j in range(0, 84):
        w.create_rectangle(40 + 5 * j, 40 + 5 * i, 45 + 5 * j, 45 + 5 * i, width=0, fill="white")
for i in range(0, 12):
    for j in range(0, 84):
        w.create_rectangle(40 + 5 * j, 40 + 5 * i, 45 + 5 * j, 45 + 5 * i, width=0, fill="gray88")
for i in range(72, 84):
    for j in range(0, 84):
        w.create_rectangle(40 + 5 * j, 40 + 5 * i, 45 + 5 * j, 45 + 5 * i, width=0, fill="gray88")
for i in range(12, 72):
    for j in range(0, 12):
        w.create_rectangle(40 + 5 * j, 40 + 5 * i, 45 + 5 * j, 45 + 5 * i, width=0, fill="gray88")
for i in range(12, 72):
    for j in range(72, 84):
        w.create_rectangle(40 + 5 * j, 40 + 5 * i, 45 + 5 * j, 45 + 5 * i, width=0, fill="gray88")
w.create_rectangle(220, 510, 280, 570, width=0, fill="white")
w.bind("<B1-Motion>", colorIn)
w.bind("<Button-2>", erase)
w.bind("<Leave>", read)

root.mainloop()
