import matplotlib.pyplot as plt
from typing import List
from scipy.optimize import curve_fit
import numpy as np

plt.rcParams["image.origin"] = 'lower'


class OOI:
    def __init__(self, name, pos_x, pos_y, inner_radius=25, outer_radius=40):
        self.name = name
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.count = [0, 0]
        self.local_map = []

    def get_pos(self, text=True):
        if text:
            return self.pos_x, self.pos_y, self.name + " is at ({},{})".format(self.pos_x, self.pos_y)

        return self.pos_x, self.pos_y

    def set_local_map(self, image):
        self.local_map = image.copy()

    def count_pixel(self, star_image, filter_val=None, new_bg_removal: bool = False):

        if filter_val is None:
            filter_val = [1, 1]

        total_counts = []
        bg_counts = []
        bg_avgs = []

        for n, image in enumerate(star_image.images):
            for c in (0,):

                img = image.data[c, :, :].copy()
                img[img < 0] = 0
                _, total_count, bg_count, bg_avg = aperture(*self.get_pos(False), *self.get_radius(), img)
                if filter_val is not None:
                    total_count /= filter_val[n]
                    bg_count /= filter_val[n]
                    bg_avg /= filter_val[n]
                total_counts.append(total_count)
                bg_counts.append(bg_count)
                bg_avgs.append(bg_avg)

        return total_counts, bg_counts, bg_avgs

    def get_count(self):
        return self.count, (self.name + ":\n", "I'-band count: {} \nR'-band count: {}".format(*self.count))

    def set_radius(self, inrad, outrad):
        self.inner_radius = inrad
        self.outer_radius = inrad + outrad

    def get_radius(self):
        return self.inner_radius, self.outer_radius

    def obj_plot(self, image, ax):
        img = image.copy()

        x_mesh, y_mesh = np.meshgrid(range(2 * self.outer_radius), range(2 * self.outer_radius))

        ax.plot_surface(x_mesh, y_mesh, img[(self.pos_y - self.outer_radius):(self.pos_y + self.outer_radius),
                                        (self.pos_x - self.outer_radius):(self.pos_x + self.outer_radius)],
                        cmap='viridis')
        ax.set_title(self.name)

    def fitting_3d(self, image):
        img = image.copy()[(self.pos_y - self.outer_radius):(self.pos_y + self.outer_radius),
              (self.pos_x - self.outer_radius):(self.pos_x + self.outer_radius)].transpose()
        mesh = np.array([[x, y] for x in range(2 * self.outer_radius) for y in range(2 * self.outer_radius)])
        mesh_grid = np.meshgrid(range(2 * self.outer_radius), range(2 * self.outer_radius))
        mask = []
        color = []

        for x in range(0, 2 * self.outer_radius):
            for y in range(0, 2 * self.outer_radius):
                if (x - self.outer_radius) ** 2 + (y - self.outer_radius) ** 2 <= self.inner_radius ** 2:
                    color.append(img[x, y] * 0.75)
                elif self.inner_radius <= (x - self.outer_radius) ** 2 + (
                        y - self.outer_radius) ** 2 <= self.outer_radius ** 2:
                    mask.append([x, y, img[x, y]])
                    color.append(img[x, y] * 0.5)
                else:
                    mask.append([x, y, img[x, y]])
                    color.append(img[x, y] * 0.1)

        mask = np.array(mask)
        fit = curve_fit(poly_sec_ord, mask[:, :2], mask[:, 2])

        """ background removal """

        fig = plt.figure(figsize=(24, 10), num=self.name + " background removal")
        fig.suptitle(self.name + " background removal")

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.scatter(mesh[:, 0], mesh[:, 1], img, c=color)
        # ax.plot_surface(*mesh_grid, img)

        ax = fig.add_subplot(1, 2, 2, projection='3d')

        fitted_val = poly_sec_ord(mesh, *fit[0]).reshape((80, 80))

        ax.scatter(mask[:, 0], mask[:, 1], mask[:, 2])
        ax.scatter(mesh[:, 0], mesh[:, 1], fitted_val, c=color, alpha=0.4)

        """ Image without background """

        fig = plt.figure(figsize=(8, 8), num=self.name + " final")
        fig.suptitle(self.name + " without background")
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(mesh[:, 0], mesh[:, 1], img - fitted_val)

        """ moffat fit """

        moffat_fit = curve_fit(moffat_2d, mesh, img.flatten() - fitted_val.flatten(), maxfev=25600,
                               p0=(40, 40, 5000, 2, 25, -10),
                               bounds=((35, 35, 0, 0, 0, -np.inf), (45, 45, np.inf, 10, np.inf, np.inf)))
        fig = plt.figure(figsize=(8, 8), num=self.name + " woBG + fit")
        fig.suptitle(self.name + " without background and fit")
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(mesh[:, 0], mesh[:, 1], img - np.array(fitted_val).reshape((80, 80)))
        ax.scatter(mesh[:, 0], mesh[:, 1], moffat_2d(mesh, *moffat_fit[0]))

        # plt.figure()
        # print("test")
        # fit21 = np.polyfit(x_mesh[0, :], img[:, 0], 1)
        # fit22 = curve_fit(gauss, x_mesh[0, :], img[:, 0] - np.poly1d(fit21)(x_mesh[0, :]), [15, 7.5, 30, -5])
        # plt.plot(x_mesh[0, :], img[:, 0])
        # plt.plot(x_mesh[0, :], np.poly1d(fit21)(x_mesh[0, :]))
        # plt.plot(x_mesh[0, :], img[:, 0] - np.poly1d(fit21)(x_mesh[0, :]))
        # plt.plot(x_mesh[0, :], gauss(x_mesh[0, :], *fit22[0]))
        # ax = fig.add_subplot(1, 2, 2, projection='3d')
        # ax.scatter(x_mesh, y_mesh, img, c=color.flatten(), cmap='viridis')

        return moffat_fit


class StarImg:
    def __init__(self, name, img_i, img_r):
        self.name: str = name
        self.images = np.array([img_i, img_r])
        self.radial: List[List[float]] = []
        self.azimutal = []
        self.objects: List[OOI] = []
        self.flux: List[float] = []
        self.wavelength: List[float] = []
        self.filter_reduction: List[float] = []

    def set_filter(self, i_filter, r_filter):
        self.filter_reduction = [i_filter, r_filter]

    def set_flux_bvg(self, b, v, g):
        self.flux = [b, v, g]

    def set_wavelength(self, i_length, rlength):
        self.wavelength = [i_length, rlength]

    def get_i_img(self):
        return self.images[0].data

    def get_r_img(self):
        return self.images[1].data

    def calc_radial_polarization(self):
        images_copy = self.images
        x, y = np.meshgrid(range(0, 1024), range(0, 1024))
        angle_phi = phi(x, y, 512, 512)

        sin_2phi = np.sin(2 * angle_phi)
        cos_2phi = np.cos(2 * angle_phi)

        for img in images_copy:
            q_phi = -img.data[1] * cos_2phi + img.data[3] * sin_2phi
            u_phi = img.data[1] * sin_2phi + img.data[3] * cos_2phi
            plt.figure()
            plt.imshow(u_phi, cmap='gray', vmin=-50, vmax=100)
            plt.show()
            self.radial.append([q_phi, u_phi])

    def add_object(self, obj: OOI):
        obj.set_local_map(self.images[0].data[0][(obj.pos_y - obj.outer_radius):(obj.pos_y + obj.outer_radius),
                          (obj.pos_x - obj.outer_radius):(obj.pos_x + obj.outer_radius)])
        self.objects.append(obj)

    def get_objects(self, text=True):
        if text:
            out = "The following objects are marked in {}:".format(self.name)
            for obj in self.objects:
                out += "\n" + obj.get_pos(True)[2]
            return self.objects, out

        return self.objects

    def show_objects(self, inner_radius, outer_radius, band='I', showPlot=False):
        a = 1000
        if band == 'I':
            img = self.images[0].data[0, :, :].copy()
        elif band == 'R':
            img = self.images[1].data[0, :, :].copy()
        else:
            raise ValueError("ABORT wrong input")

        img[img < 0] = 0
        img = np.log(a * img + 1) / np.log(a)
        for obj in self.objects:
            obj.set_radius(inner_radius, outer_radius)
            img, _, _, _ = aperture(*obj.get_pos(False), inner_radius, outer_radius, img)

        if showPlot:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(img)

            plt.subplot(1, 2, 2)
            plt.imshow(np.log(self.images[0].data[0, :, :] + 1) / np.log(a))
            plt.tight_layout()
        return img

    def azimuthal_fitting(self, image: str):
        if image.lower() == "i":
            img = self.images[0].data[0].copy()
        elif image.lower() == "r":
            img = self.images[1].data[0].copy()
        else:
            raise ValueError("Wrong image name")
        profile = azimuthal_averaged_profile(img)

        plt.figure()
        plt.title(self.name + " Azimuthal profile and fit")
        plt.semilogy(range(0, 512), profile)

        try:
            fitting = curve_fit(moffat_1d, range(0, 450), profile[:450],
                                bounds=((0, 0, 0, 0, -np.inf), (5, np.inf, np.inf, np.inf, np.inf)))
            print("Azimuthal profil fitting paramter:")
            print(fitting[0])
            plt.semilogy(range(0, 512), moffat_1d(np.arange(0, 512), *fitting[0]))
        except RuntimeError:
            print("The fitting didnt work :(")
            fitting = None

        return np.array([np.arange(0, 512), np.array(profile)]), fitting


def azimuthal_averaged_profile(image: np.ndarray):
    size = image[0].size
    radius = size//2
    cx, cy = size//2, size//2
    x, y = np.arange(0, 2 * radius), np.arange(0, 2 * radius)
    img = image.copy()
    profile = []
    for r in range(0, radius):
        mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 <= r ** 2
        ring = img * mask

        profile.append(np.average(ring[ring != 0]))

        img[mask] = 0

    return np.array(profile)


def poly_sec_ord(pos, x0, y0, axx, ayy, axy, bx, by, c):
    return axx * (pos[:, 0] - x0) ** 2 + ayy * (pos[:, 1] - y0) ** 2 + axy * (pos[:, 0] - x0) * (
            pos[:, 1] - y0) + bx * (pos[:, 0] - x0) + by * (pos[:, 1] - y0) + c


def moffat_1d(x, x0, alpha, beta, gamma, b, offset=True):
    if not offset:
        b = 0
    return alpha * (1 + (x - x0) ** 2 / gamma) ** (-beta) + b


def moffat_2d(coord, x0, y0, alpha, beta, gamma, b):
    x = coord[:, 0]
    y = coord[:, 1]
    return alpha * (1 + ((x - x0) ** 2 + (y - y0) ** 2) / gamma) ** (-beta) + b


def gauss(pos, a, sigma, mu, b):
    return a * np.exp(-(pos - mu) ** 2 / (2 * sigma ** 2)) + b


def ballestero(bv):
    return 4600 * (1 / (0.92 * bv + 1.7) + 1 / (0.92 * bv + 0.62))


def phi(x, y, x0, y0):
    return np.arctan((x - x0) / (y - y0))


def aperture(pos_x, pos_y, inner_radius, outer_radius, img):
    inner_radius = int(inner_radius)
    outer_radius = inner_radius + int(outer_radius)
    obj_count = 0
    obj_pixel = 0
    background_count = 0
    background_pixel = 0
    img_copy = img.copy()
    (size_y, size_x) = img.shape

    if not ((pos_x + outer_radius) < size_x and (pos_x - outer_radius) > 0):
        raise ValueError("error x axis")
    if not ((pos_y + outer_radius) < size_y and (pos_y - outer_radius) > 0):
        raise ValueError("error y axis")

    for x in range(0, 2 * outer_radius + 1):
        for y in range(0, 2 * outer_radius + 1):
            if (x - outer_radius) ** 2 + (y - outer_radius) ** 2 <= inner_radius ** 2:
                img_copy[pos_y - outer_radius + y, pos_x - outer_radius + x] *= 0.99
                obj_count += img[pos_y - outer_radius + y, pos_x - outer_radius + x]
                obj_pixel += 1
            if inner_radius ** 2 < (x - outer_radius) ** 2 + (y - outer_radius) ** 2 <= outer_radius ** 2:
                img_copy[pos_y - outer_radius + y, pos_x - outer_radius + x] *= 0.8
                background_count += img[pos_y - outer_radius + y, pos_x - outer_radius + x]
                background_pixel += 1

    if background_pixel != 0:
        background_avg = background_count / background_pixel
    else:
        background_avg = 0

    obj_wo_background = obj_count - background_avg * obj_pixel

    return img_copy, obj_count, obj_wo_background, background_avg
