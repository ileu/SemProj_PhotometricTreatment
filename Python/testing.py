import numpy as np
import matplotlib.pyplot as plt
from StarFunctions import aperture, azimuthal_averaged_profile
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


test = np.full((2 + 1, 2 + 1, 2 + 1, 2), 1)
test2 = np.full((2, 2 + 1, 2 + 1, 2 + 1), 1)
test[..., 1] = 2
test2[1] = 2
print(test[0, 0, 0, 0])
print(tuple(range(test.ndim - 1)))
print(np.mean(test, axis=(0, 1, 2)))
print(np.mean(test2[0]), np.mean(test2[1]))
size = 150

cyc116_i = np.array([np.sum(cyc116.get_i_img()[0][aperture((1024, 1024), 512, 512, r)]) for r in range(1, 512)])
x, profile, _ = cyc116.azimuthal[0]
circumference = [np.sum(aperture((1024, 1024), 512, 512, r, r - 1)) for r in range(1, 513)]

summing = np.array([np.sum((profile * circumference)[:r]) for r in range(1, 512)])

print(cyc116_i[:10])
print(summing[:10])
azimuthal_averaged_profile(cyc116.get_i_img()[0])

for r in range(size // 2):
    test = aperture((150, 150), 75, 75, r, r - 1) * aperture((150, 150), 75, 75, size // 2, r + 1)
    if not np.all(test == 0):
        print("UPSI")
