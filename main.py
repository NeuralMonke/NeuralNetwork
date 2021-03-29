import numpy as np
import scipy.special


class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.wih = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.who = np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))
        self.learning_rate = learning_rate
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        train_inputs = np.array(inputs_list, ndmin=2).T
        train_targets = np.array(targets_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, train_inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = train_targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        self.who += self.learning_rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                                np.transpose(hidden_outputs))
        self.wih += self.learning_rate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                                np.transpose(train_inputs))
        pass

    def query(self, inputs_list):
        _inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, _inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


INPUT_NODES = 784
HIDDEN_NODES = 200
OUTPUT_NODES = 10
LEARNING_RATE = 0.1
EPOCHS = 5

n = NeuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)

training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

for e in range(EPOCHS):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(OUTPUT_NODES) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

test_data_file = open("mnist_test.csv", 'r')
test_data_list = training_data_file.readlines()
test_data_file.close()

scorecard = []

for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)
    pass
pass

scorecard_array = np.asarray(scorecard)
print("Эффективность: ", scorecard_array.sum() / scorecard_array.size)
