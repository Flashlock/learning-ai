import numpy as np
import matplotlib.pyplot as plt

class Perceptron(object):
    """
    The Perceptron Algorithm:
    
    Parameters:
        omega - a vector of length of num axes + 1. each element starts randomized and is updated
            to converge to the separation

        nu - learning rate. Constant between 0 and 1. Bigger nu's will 
            converge to the separation faster, but make more mistakes.

    Inputs:
        x - a point represented by a vector of length of num axes + 1. 
            x0 = 1.
            x1 to xn represent the coordinates for this point.
            there will be an array of x's inputed
    """


    # constant multiplier
    learning_rate: float
    # how many times to iterate over dataset
    iterations: int
    # initial states are randomized to this seed
    start_seed: int

    # weights after fitting
    weights: np.array
    # misclassifications after fitting
    errors: list


    def __init__(self, learning_rate=.01, iterations=50, start_seed=1):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.start_seed = start_seed

    def fit(self, training_vectors, target_values):
        # training vectors is the array of x's
        # target values are what the x's should be identified as
        # they must be of the same length
        assert len(training_vectors) == len(target_values)

        # create a random number generator given the seed
        rngen = np.random.RandomState(self.start_seed)
        # get the weights randomly along a bell curve.
        # loc: distribution center, scale: standard deviation, size: number of samples to return
        self.weights = rngen.normal(loc=0.0, scale=.01, size=1 + training_vectors.shape[1])

        self.errors = list()
        # now loop through the vectors, make predictions, log mistakes
        for iteration in range(0, self.iterations):
            errors = 0
            for x, target in zip(training_vectors, target_values):
                # make the prediction
                prediction = self.predict(x)
                # update the weights
                update = self.learning_rate * (target - prediction)
                # increment all the weights - update will be 0 if prediction correct
                self.weights[1:] += update * x
                self.weights[0] += update

                # count the number of errors
                errors += 1 if update != 0 else 0

            self.errors.append(errors)
            # break for perfection
            if(errors == 0):
                break

    # determines the net input for some x
    def net_input(self, x):
        return np.dot(x, self.weights[1:]) + self.weights[0]

    # makes a prediction for some x
    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)



perceptron = Perceptron()
training_vectors=np.array([[1, 10],[3, 20], [5, 25], [-1, 15], [-3, -20], [-5, -24]])
target_values=np.array([1, 1, 1, -1, -1, -1])
perceptron.fit(training_vectors, target_values)

x = [coord[0] for coord in training_vectors]
y = [coord[1] for coord in training_vectors]
plt.scatter(x, y)

p_line_x = list()
p_line_y = list()
for i in range(-3, 3):
    p_line_x.append(i)
    y = (perceptron.weights[0] + perceptron.weights[1] * i) / perceptron.weights[2]
    p_line_y.append(y)

plt.scatter(p_line_x, p_line_y, color='red')

plt.savefig('test.png', bbox_inches='tight')