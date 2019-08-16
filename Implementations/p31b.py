import ANN
import numpy as np
from random import shuffle

s = ANN.ANN([13,8,5,3])
dataset = ANN.load_data('Drinks.csv')

a = 70

inputs = []
outputs = []
shuffle(dataset)
for data in dataset:
	data = list(map(float, data))
	y = data[13:]
	outputs.append(y)
	data = data[:13]
	inputs.append(data)

inputsTrain = inputs[:-a]
inputsTest = inputs[-a:]

outputsTrain = outputs[:-a]
outputsTest = outputs[-a:]
	# print(data)
# print(np.matmul([1,2],[[2],[3]]))
# s.forward_propagate(inputs[0])
# print(outputs)
# s.backward_propagete(outputs[0])
# s.update_weights()
s.train_ANN(inputsTrain, outputsTrain)
# s.print_parameters()
s.accuracy(inputsTest,outputsTest)
