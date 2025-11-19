import numpy
import scipy.special
import matplotlib.pyplot


class neuralNetwork:
    def __init__(self, inputnodes, hiddennods, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennods
        self.onodes = outputnodes
        self.lr = learningrate

        self.weight_input_hid = (
            numpy.random.rand(self.hnodes, self.inodes)-0.5)

        self.weight_hid_output = (
            numpy.random.rand(self.onodes, self.hnodes)-0.5)

        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, target_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.weight_input_hid, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.weight_hid_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.weight_hid_output.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.weight_hid_output += self.lr * \
            numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),
                      numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.weight_input_hid += self.lr * \
            numpy.dot((hidden_errors*hidden_outputs *
                      (1-hidden_outputs)), numpy.transpose(inputs))

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.weight_input_hid, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.weight_hid_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
training_data_file = open(
    "makeyourownneuralnetwork/mnist_dataset/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
for record in training_data_list:
    all_values = record.split(',')
    scaled_input = (numpy.asarray(all_values[1:], dtype=float)/255.0*0.99)+0.01
    targets = numpy.zeros(output_nodes)+0.01
    targets[int(all_values[0])] = 0.99
    n.train(scaled_input, targets)

test_data_file = open(
    "makeyourownneuralnetwork/mnist_dataset/mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

test_all_values = test_data_list[1].split(',') # 7
print(test_all_values[0])
test_image_array = numpy.asarray(test_all_values[1:], dtype=float).reshape((28, 28))

print(n.query((numpy.asarray(test_all_values[1:], dtype=float)/255.0*0.99)+0.01))

matplotlib.pyplot.imshow(test_image_array,cmap='Greys',interpolation="None")
matplotlib.pyplot.show() 


