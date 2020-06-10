from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt

size = 100


def angle_phi(x, y, x0, y0):
    results = np.true_divide(x - x0, y - y0, out=np.zeros((size, size)), where=(x != size // 2) | (y != size // 2))
    return np.arctan(results)


def angle_phi2(x, y, x0, y0):
    size = len(y)
    out = np.zeros((size, size))
    out[:, :size // 2] = -np.inf
    out[:, size // 2 + 1:] = np.inf
    results = np.true_divide(x - x0, y - y0, out=out, where=(x == x) & (y != size // 2))
    return np.arctan(results)


y, x = np.ogrid[:size, :size]
radius = np.sqrt((x - size // 2) ** 2 + (y - size // 2) ** 2)
phi = angle_phi(x, y, size // 2, size // 2)
phi2 = angle_phi2(x, y, size // 2, size // 2)
mask = (x == x) & (y != size // 2)
print(phi[size // 2, 0])
print(phi2[size // 2, 0])
plt.figure()
plt.title("Phi")
plt.imshow(phi, origin='lower')
plt.figure()
plt.title("Phi2")
plt.imshow(phi2, origin='lower')
plt.figure()
plt.title("blalba")
plt.imshow(mask, origin='lower')

plt.show()

test = np.zeros((size, size))

print(test[0, :])
print(test[0, size // 2])
