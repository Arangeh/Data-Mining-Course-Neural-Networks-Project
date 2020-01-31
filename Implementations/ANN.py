import numpy as np
import csv


def load_data(filename):
	"""
	Loads transactions from given file
	:param filename:
	:return:
	"""
	trans = []
	reader = csv.reader(open(filename, 'r'), delimiter=',')
	for row in reader:
		if (row[-1] == "Class 3"):
			continue
		trans.append(row)
	return trans


def softmax(x):
	"""Compute softmax values for each sets of scores in x."""
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum(axis=0)  # only difference


class ANN:
	def __init__(self, a):
		self.layerSizes = a
		self.layers = []
		self.layersRaw = []
		self.w = []
		self.delta = []
		self.len = len(a)
		for i in range(len(a)):
			self.layers.append([0.0] * a[i])
			self.layersRaw.append([0.0] * a[i])
			if (i != 0):
				self.delta.append([0.0] * a[i])
			
			# It can be optimized in order to use less memory
			# self.w.append(np.ones(shape=(a[i + 1], a[i])) if (i != len(a) - 1) else np.ones(shape=(1, a[i])))
			if (i != len(a) - 1):
				self.w.append(np.random.rand(a[i + 1], a[i]))
	
	# return self.layers
	
	def kronecker_delta(self, m, n):
		if (m == n):
			return 1
		else:
			return 0
	
	# The 'input' parameter determines the inputs as the first layer of the ANN.Then we propagate through the next layers
	# of the network. Untill we calculate the final output layer based on previous set of layers and weights
	def forward_propagate(self, input):
		self.layers[0] = input
		self.layersRaw[0] = input
		for i in range(self.len - 1):
			next = np.matmul(self.w[i], np.transpose(self.layers[i]))
			self.layersRaw[i + 1] = next
			self.layers[i + 1] = np.tanh(next) if i != self.len - 2 else softmax(next)
		self.print_parameters()
	
	def backward_propagate(self, outputs):
		l = self.len
		for i in range(l - 2, -1, -1):
			if (i == l - 2):
				# y_k - t_k
				sum = 0
				deltaHelp = [a - b for a, b in zip(self.layers[i + 1], outputs)]
				# print(deltaHelp)
				for m in range(len(self.delta[i])):
					for n in range(len(outputs)):
						sum += deltaHelp[n] * self.layers[i + 1][n] * (self.kronecker_delta(m, n) - self.layers[i + 1][m])
					self.delta[i][m] = sum
					sum = 0
			else:
				for j in range(len(self.delta[i])):
					sum = 0
					for k in range(len(self.layers[i + 2])):
						sum += self.w[i + 1][k][j] * self.delta[i + 1][k]
					sum = sum * (
							1.0 - (self.layers[i + 1][j] * self.layers[i + 1][j]))  # since derivative of tanh(x) = 1 - tanh(x)^2
					self.delta[i][j] = sum
	
	def update_weights(self):
		for i in range(self.len - 1):
			for j in range(len(self.layers[i + 1])):
				for k in range(len(self.layers[i])):
					self.w[i][j][k] += -0.7 * self.delta[i][j] * self.layers[i][k]
	
	def train_ANN(self, inputs, outputs):
		for i in range(len(inputs)):
			self.forward_propagate(inputs[i])
			self.backward_propagate(outputs[i])
			self.update_weights()
	
	def accuracy(self, inputs, outputs):
		total_errors = 0
		for i in range(len(inputs)):
			self.forward_propagate(inputs[i])
			# print(i)
			# print("#")
			# print(outputs[i])
			# print(self.layers[self.len - 1])
			# print(np.argmax(self.layers[self.len - 1]))
			argmax = np.argmax(self.layers[self.len - 1])
			if (outputs[i][argmax] != 1.0):  # misclassification
				total_errors += 1
			
			print("Total # of misclassified items: " + total_errors.__str__())
			accuracy = 1 - float(total_errors) / float(len(inputs))
			print("Accuracy = " + accuracy.__str__())
	
	def print_parameters(self):
		print("Optimum Weights found so far:")
		for i in range(self.len - 1):
			print(i)
			print(self.w[i])
		
		print("Input Values: ")
		print(self.layers[0])
		
		print("Output Values: ")
		print(self.layers[self.len - 1])
