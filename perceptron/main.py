import cleaner
import perceptron
import matplotlib.pyplot as plt
import numpy as np

data = cleaner.get_data('../datasets/iris/iris.data')
perc = perceptron.Perceptron()

# scatter plot the data
plt.figure(0)
plt.scatter(data[0][:50, 0], data[0][:50, 1], color='red', marker='o', label='setosa')
plt.scatter(data[0][50:, 0], data[0][50:, 1], color='blue', marker='v', label='versicolor')
plt.legend(loc='upper left')
plt.title('Untrained Data')
plt.savefig('data.png')

# fit to data, plot error convergence
plt.figure(1)
perc.fit(data[0], data[1]) 
plt.plot(range(1, len(perc.errors) + 1), perc.errors)
plt.xlabel('Epochs')
plt.ylabel('Number of Updates')
plt.title('Error Convergence')
plt.savefig('errors.png')

# plot separation
plt.figure(2)
increments = [i for i in np.arange(3, 8, .25)]
increments = np.array(increments)
ys = -(perc.weights[1] * increments[:] + perc.weights[0]) / perc.weights[2]
plt.plot(increments, ys)
plt.scatter(data[0][:50, 0], data[0][:50, 1], color='red', marker='o', label='setosa')
plt.scatter(data[0][50:, 0], data[0][50:, 1], color='blue', marker='v', label='versicolor')
plt.legend(loc='upper left')
plt.title('Trained Data')
plt.savefig('trained.png')
