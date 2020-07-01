import numpy as np
import matplotlib.pyplot as plt
from StarFunctions import aperture, angle_phi, azimuthal_averaged_profile
import time
from StarData import cyc116


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


cyc116_i = np.array([np.sum(cyc116.get_i_img()[0][aperture((1024, 1024), 512, 512, r)]) for r in range(1, 512)])
x, profile, _ = cyc116.azimuthal[0]
circumference = [np.sum(aperture((1024, 1024), 512, 512, r, r - 1)) for r in range(1, 513)]

summing = np.array([np.sum((profile * circumference)[:r]) for r in range(1, 512)])

print(cyc116_i[:10])
print(summing[:10])
azimuthal_averaged_profile(cyc116.get_i_img()[0])

size = 150
for r in range(size // 2):
    test = aperture((150, 150), 75, 75, r, r - 1) * aperture((150, 150), 75, 75, size // 2, r + 1)
    if not np.all(test == 0):
        print("UPSI")

x, y = np.meshgrid(range(0, size), range(0, size))

distance = np.sqrt((x - size // 2) ** 2 + (y - size // 2) ** 2)
mask1 = np.where((0 <= distance) & (distance < 25), 0.5, 1)
mask2 = np.where((25 <= distance) & (distance < 50), 0, 1)

plt.imshow(mask1 + mask2, cmap='Set1')
plt.show()

"""
size = 100
min_size = 50
legnth = 200
img = np.full((legnth, legnth), 5)

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
"""
