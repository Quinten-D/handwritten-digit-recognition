import numpy as np


class NeuralNetwork:
    def __init__(self, net_layout):
        self.L = len(net_layout) - 1
        self.structure = net_layout
        self.weights = []
        self.a = []
        self.z = []
        self.delta = []
        self.gradient = []
        for i in range(0, self.L):
            self.weights.append(np.random.randn(self.structure[i+1], self.structure[i]))
            self.a.append(np.zeros((self.structure[i+1], 1)))
            self.z.append(np.zeros((self.structure[i + 1], 1)))
            self.delta.append(np.zeros((self.structure[i + 1], 1)))

    def resetGradient(self):
        self.gradient = []
        for i in range(0, self.L):
            self.gradient.append(np.zeros((self.structure[i + 1], self.structure[i])))

    def feedforward(self, input):
        self.z[0] = np.dot(self.weights[0], input)
        self.a[0] = sigmoid(self.z[0])
        for counter in range(1, self.L):
            self.z[counter] = np.dot(self.weights[counter], self.a[counter - 1])
            self.a[counter] = sigmoid(self.z[counter])

    def backpropagation(self, training_example):
        self.delta[-1] = (-training_example[1] + self.a[-1]) * derivativeSigmoid(self.z[-1])
        self.gradient[-1] += (1/10) * np.dot(self.delta[-1], self.a[-2].transpose())
        for min_counter in range(2, self.L):
            counter = -min_counter
            self.delta[counter] = np.dot(self.weights[counter+1].transpose(), self.delta[counter+1]) * derivativeSigmoid(self.z[counter])
            self.gradient[counter] += (1/10) * np.dot(self.delta[counter], self.a[counter-1].transpose())
        self.delta[0] = np.dot(self.weights[1].transpose(), self.delta[1]) * derivativeSigmoid(self.z[0])
        self.gradient[0] += (1/10) * np.dot(self.delta[0], training_example[0].transpose())

    def stochasticGradientDescent(self):
        for i in range(0, self.L):
            self.weights[i] -= 3 * self.gradient[i]

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = []
        for (x, y) in test_data:
            self.feedforward(x)
            test_results.append((np.argmax(self.a[-1]), y))
        return sum(int(x == y) for (x, y) in test_results)


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def derivativeSigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))