import numpy
import matplotlib.pyplot


data_file = open("makeyourownneuralnetwork/mnist_dataset/mnist_train_100.csv",'r')
data__list = data_file.readlines()
data_file.close()
print(len(data__list))
print(data__list[0].count(",")) #28*28

all_values = data__list[1].split(',')

scaled_input = (numpy.asarray(all_values[1:], dtype=float)/255.0*0.99)+0.01
# print(scaled_input)


onodes = 10
targets = numpy.zeros(onodes)+0.01
targets[int(all_values[0])] = 0.99
print(targets)


image_array = numpy.asarray(all_values[1:], dtype=float).reshape((28, 28))


matplotlib.pyplot.imshow(image_array,cmap='Greys',interpolation="None")
matplotlib.pyplot.show() 