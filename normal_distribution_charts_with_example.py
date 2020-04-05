import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def contour(plt, score, x_axis, y_axis, colour, lessthan=True):
    #plt.axvline(score, color=colour)
    x_temp = [x for x in x_axis if x <= score]
    if lessthan:
        y_temp = y_axis[:len(x_temp)]
    else:
        y_temp = y_axis[len(x_temp): ]
        x_temp = [x for x in x_axis if x > score]
    plt.fill_between(x_temp, 0, y_temp, facecolor=colour)

if __name__ == '__main__':
    mean = 80; sd = 15
    x_axis = np.arange(10, 140, 0.001)
    y_axis = norm.pdf(x_axis,mean,sd)

    plt.plot(x_axis, y_axis)
    plt.xlabel('Scores')
    plt.ylabel('PDF')
    plt.title("Gaussian Distribution with Mean: {} and STD: {}".format(mean, sd))

    colour='#4dac26'
    score = 60
    contour(plt, score, x_axis, y_axis, colour, lessthan=True)

    colour='#f1b6da'
    score = 90
    contour(plt, score, x_axis, y_axis, colour, lessthan=False)

    plt.show()
