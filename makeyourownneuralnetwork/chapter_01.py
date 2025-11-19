# how Neural Network work
import math

w11 = 0.9
w12 = 0.2
w21 = 0.3
w22 = 0.8
input1 = 1
input2 = 0.5

x1 = input1*w11 + input2*w21
x2 = input1*w12 + input2*w22
y1 = 1/(1 + 1/((math.e)**x1))
y2 = 1/(1 + 1/((math.e)**x2))
print(y1, y2)

