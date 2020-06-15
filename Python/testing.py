from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
from StarFunctions import annulus, angle_phi
import time


def f_numpy(names, values):
    result_names = np.unique(names)
    print(result_names)
    result_values = np.empty(result_names.shape)

    for i, name in enumerate(result_names):
        result_values[i] = np.mean(values[names == name])

    return result_names, result_values


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2 - time1) * 1000.0))

        return ret

    return wrap


size = 100
min_size = 50
legnth = 200
img = np.full((legnth, legnth), 5)
x, y = np.meshgrid(range(0, legnth), range(0, legnth))
test = annulus(img, np.inf, min_size)
angles = angle_phi(x, y, size, size)
distance = np.sqrt((x - size) ** 2 + (y - size) ** 2)
coordinates = np.column_stack((distance, angles))
mask = np.where((-1.58 < angles) & (angles <= -0.78), 1, 0)

print(np.min(distance[mask]))

plt.figure()
plt.imshow(distance*mask)
plt.show()

result = f_numpy(np.rint(distance*mask), img*mask)
print(result)

spacing, ds = np.linspace(-1.58, 1.58, 4, retstep=True, endpoint=False)
plt.figure()
plt.imshow(img * test)


for space in spacing:
    mask = np.where((space < angles) & (angles <= space + ds), 1, 0)
    plt.figure()
    plt.imshow(mask)
plt.show()
