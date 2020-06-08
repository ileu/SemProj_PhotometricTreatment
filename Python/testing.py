import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
from StarFunctions import ballestero

data = np.array([[-0.33, -0.32],
                 [-0.3, -0.29],
                 [-0.02, -0.02],
                 [0.3, 0.17],
                 [0.58, 0.31],
                 [0.81, 0.42],
                 [1.4, 0.91]])

fit = np.polyfit(data[:, 1], data[:, 0], 1)
p = np.poly1d(fit)
plt.scatter(data[:, 1], data[:, 0])
plt.plot(data[:, 1], p(data[:, 1]))
plt.show()
