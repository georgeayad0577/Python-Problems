import numpy as np
from sklearn.linear_model import Perceptron

class NeuralNetwork():

    def __init__(self,learning_rate,threshold):
        # seeding for random number generation
        self.learning_rate=learning_rate
        self.threshold=threshold
        np.random.seed(1)
        #intialize weights
        # converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def step(self, x):
        # applying the step function

        if x>float(self.threshold):
            return 1
        else :
            return 0


    def train(self, training_inputs, training_outputs, training_iterations):
        # training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            # siphon the training data via  the neuron
            output = self.think(training_inputs)

            # computing error rate for back-propagation
            error = training_outputs - output

            # performing weight adjustments
            adjustments = np.dot(training_inputs.T, error * self.learning_rate)

            self.synaptic_weights += adjustments

    def think(self, inputs):
        # passing the inputs via the neuron to get output
        # converting values to floats

        inputs = inputs.astype(float)# convert input to float datatype
        output_in = np.sum(np.dot(inputs, self.synaptic_weights))#weighted sum of input
        output=self.step(output_in)
        return output


if __name__ == "__main__":
    # initializing the neuron class
    learning_rate=float(input("Learning  rate: "))
    threshold=float(input("Threshold: "))
    neural_network = NeuralNetwork(learning_rate,threshold)

    print("Beginning Randomly Generated Weights: ")
    print(neural_network.synaptic_weights)

    # training data consisting of 4 examples--3 input values and 1 output
    training_inputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1],
                                [0, 0, 0]])

    training_outputs = np.array([[1, 1, 1, 1,0]]).T

    # training taking place
    neural_network.train(training_inputs, training_outputs, 4)

    print("Ending Weights After Training: ")
    print(neural_network.synaptic_weights)

    user_input_one = str(input("User Input One: "))
    user_input_two = str(input("User Input Two: "))
    user_input_three = str(input("User Input Three: "))

    print("Considering New Situation: ", user_input_one, user_input_two, user_input_three)
    print("New Output data: ")
    print(neural_network.think(np.array([user_input_one, user_input_two, user_input_three])))
    print("Wow, we did it!")

x = np.array([[0, 0, 1],
              [1, 1, 1],
              [1, 0, 1],
              [0, 1, 1],
              [0, 0, 0]])

y = np.array([[1, 1, 1, 1,0]]).T

model = Perceptron(random_state = 1)
m = model.fit(x, y)

print(m.score([[1,0,1]],[1]))