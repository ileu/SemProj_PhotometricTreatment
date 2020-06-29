import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List
from scipy.optimize import curve_fit, least_squares
from scipy import interpolate
import numpy as np
import mayavi.mlab as mlab
import pickle
import os

plt.rcParams["image.origin"] = 'lower'
full_file_path = os.getcwd()


class OOI:
    def __init__(self, name, pos_x, pos_y):
        self.name = name
        self.pos_x = pos_x
        self.pos_y = pos_y

    def get_pos(self, text=True):
        if text:
            return self.pos_x, self.pos_y, self.name + " is at ({},{})".format(self.pos_x, self.pos_y)

        return self.pos_x, self.pos_y

    # def fitting_3d(self, irad, orad, image):
    #     img = image.copy()[(self.pos_y - orad):(self.pos_y + orad),
    #           (self.pos_x - orad):(self.pos_x + orad)].transpose()
    #     mesh = np.array([[x, y] for x in range(2 * orad) for y in range(2 * orad)])
    #     mesh_grid = np.meshgrid(range(2 * orad), range(2 * orad))
    #     mask = []
    #     color = []
    #
    #     for x in range(0, 2 * orad):
    #         for y in range(0, 2 * orad):
    #             if (x - orad) ** 2 + (y - orad) ** 2 <= irad ** 2:
    #                 color.append(img[x, y] * 0.75)
    #             elif irad <= (x - orad) ** 2 + (
    #                     y - orad) ** 2 <= orad ** 2:
    #                 mask.append([x, y, img[x, y]])
    #                 color.append(img[x, y] * 0.5)
    #             else:
    #                 mask.append([x, y, img[x, y]])
    #                 color.append(img[x, y] * 0.1)
    #
    #     mask = np.array(mask)
    #     fit = curve_fit(poly_sec_ord, mask[:, :2], mask[:, 2])
    #
    #     """ background removal """
    #
    #     fig = plt.figure(figsize=(24, 10), num=self.name + " background removal")
    #     fig.suptitle(self.name + " background removal")
    #
    #     ax = fig.add_subplot(1, 2, 1, projection='3d')
    #     ax.scatter(mesh[:, 0], mesh[:, 1], img, c=color)
    #     # ax.plot_surface(*mesh_grid, img)
    #
    #     ax = fig.add_subplot(1, 2, 2, projection='3d')
    #
    #     fitted_val = poly_sec_ord(mesh, *fit[0]).reshape((80, 80))
    #
    #     ax.scatter(mask[:, 0], mask[:, 1], mask[:, 2])
    #     ax.scatter(mesh[:, 0], mesh[:, 1], fitted_val, c=color, alpha=0.4)
    #
    #     mlab.figure(figure=self.name + " background removal")
    #     mlab.points3d(mask[:, 0], mask[:, 1], mask[:, 2], scale_mode="none", scale_factor=1)
    #     mlab.points3d(mesh[:, 0], mesh[:, 1], fitted_val.flatten(), color, scale_mode="none", scale_factor=1)
    #
    #     """ Image without background """
    #
    #     fig = plt.figure(figsize=(8, 8), num=self.name + " final")
    #     fig.suptitle(self.name + " without background")
    #     ax = fig.add_subplot(1, 1, 1, projection='3d')
    #     ax.scatter(mesh[:, 0], mesh[:, 1], img - fitted_val)
    #
    #     mlab.figure(figure=self.name + " final")
    #     mlab.surf(img - fitted_val)
    #
    #     """ moffat fit """
    #
    #     moffat_fit = curve_fit(moffat_2d, mesh, img.flatten() - fitted_val.flatten(), maxfev=25600,
    #                            p0=(40, 40, 5000, 2, 25, -10),
    #                            bounds=((35, 35, 0, 0, 0, -np.inf), (45, 45, np.inf, 10, np.inf, np.inf)))
    #     fig = plt.figure(figsize=(8, 8), num=self.name + " woBG + fit")
    #     fig.suptitle(self.name + " without background and fit")
    #     ax = fig.add_subplot(1, 1, 1, projection='3d')
    #     ax.scatter(mesh[:, 0], mesh[:, 1], img - fitted_val)
    #     ax.scatter(mesh[:, 0], mesh[:, 1], moffat_2d(mesh, *moffat_fit[0]))
    #
    #     mlab.figure(figure=self.name + " woBG + fit")
    #     mlab.points3d(mesh[:, 0], mesh[:, 1], (img - fitted_val).flatten(), color=(0, 0, 1), scale_mode="none",
    #                   scale_factor=1.75)
    #     mlab.points3d(mesh[:, 0], mesh[:, 1], moffat_2d(mesh, *moffat_fit[0]), color=(1, 165 / 255, 0),
    #                   scale_mode="none", scale_factor=1.75)
    #
    #     # plt.figure()
    #     # print("test")
    #     # fit21 = np.polyfit(x_mesh[0, :], img[:, 0], 1)
    #     # fit22 = curve_fit(gauss, x_mesh[0, :], img[:, 0] - np.poly1d(fit21)(x_mesh[0, :]), [15, 7.5, 30, -5])
    #     # plt.plot(x_mesh[0, :], img[:, 0])
    #     # plt.plot(x_mesh[0, :], np.poly1d(fit21)(x_mesh[0, :]))
    #     # plt.plot(x_mesh[0, :], img[:, 0] - np.poly1d(fit21)(x_mesh[0, :]))
    #     # plt.plot(x_mesh[0, :], gauss(x_mesh[0, :], *fit22[0]))
    #     # ax = fig.add_subplot(1, 2, 2, projection='3d')
    #     # ax.scatter(x_mesh, y_mesh, img, c=color.flatten(), cmap='viridis')
    #
    #     return moffat_fit


class StarImg:
    def __init__(self, name, img_i, img_r):
        self.name: str = name
        self.images = np.array([img_i, img_r])
        self.disk = None
        self.radial: np.ndarray = []
        self.half_azimuthal = []
        self.azimuthal = []
        self.azimuthal_qphi = []
        self.objects: List[OOI] = []
        self.flux: List[float] = []
        self.wavelength: List[float] = []
        self.filter_reduction: List[float] = []

    def save(self):
        print(self.name)
        self.calc_radial_polarization()
        for index, img in enumerate(self.images):
            self.azimuthal.append(azimuthal_averaged_profile(img.data[0], err=True))
            self.half_azimuthal.append(half_azimuthal_averaged_profile(img.data[0], err=True))
            self.azimuthal_qphi.append(azimuthal_averaged_profile(self.radial[index][0], err=True))

        save = [self.radial, self.azimuthal, self.half_azimuthal, self.azimuthal_qphi]

        pickle.dump(save, open(full_file_path + "/../Data/" + self.name + "_save.p", "wb"))
        print("File saved")

    def load(self):
        [self.radial, self.azimuthal, self.half_azimuthal, self.azimuthal_qphi] = pickle.load(
            open(full_file_path + "/../Data/" + self.name + "_save.p", "rb"))
        print("File loaded")

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

    def set_disk(self, disk):
        self.disk = disk

    def calc_radial_polarization(self):
        images_copy = self.images
        x, y = np.meshgrid(range(0, 1024), range(0, 1024))
        phi = angle_phi(x, y, 512, 512)

        sin_2phi = np.sin(2 * phi)
        cos_2phi = np.cos(2 * phi)
        radial = []

        for img in images_copy:
            q_phi = -img.data[1] * cos_2phi + img.data[3] * sin_2phi
            u_phi = img.data[1] * sin_2phi + img.data[3] * cos_2phi
            radial.append([q_phi, u_phi])

        self.radial = np.array(radial)

    def add_object(self, obj: OOI):
        self.objects.append(obj)

    def get_objects(self, text=True):
        if text:
            out = "The following objects are marked in {}:".format(self.name)
            for obj in self.objects:
                out += "\n" + obj.get_pos(True)[2]
            return self.objects, out

        return self.objects

    def mark_disk(self, inner_radius, middle_radius, outer_radius, band='I', err=False):
        if self.disk is None:
            raise ValueError("Please assign a disk first")

        total_counts = []
        wo_bg_counts = []
        background_avgs = []
        error = []

        radial_copy: np.ndarray = self.radial.copy()
        frame, phi, h, w = self.radial.shape

        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((x - self.disk.pos_x) ** 2 + (y - self.disk.pos_y) ** 2)
        radius = np.array(radius)

        mask1 = (inner_radius <= radius) & (radius <= middle_radius)
        obj_pixel = np.count_nonzero(mask1)

        mask2 = (middle_radius < radius) & (radius <= outer_radius)
        background_pixel = np.count_nonzero(mask2)

        for img in radial_copy:
            total_counts.append(np.sum(img[0, mask1]))
            background_avgs.append(np.sum(img[0, mask2]) / background_pixel)
            wo_bg_counts.append(total_counts[-1] - background_avgs[-1] * obj_pixel)
            if err:
                error.append([np.sqrt(total_counts[-1]),
                              np.sqrt(total_counts[-1] + (np.std(img[0, mask2]) / background_pixel * obj_pixel) ** 2)])

        if band == "I":
            radial_copy[0, 0][mask1] *= 3
            radial_copy[0, 0][mask2] *= 5
            if err:
                return np.array(radial_copy[0, 0]), np.array(total_counts), np.array(wo_bg_counts), np.array(
                    background_avgs), np.array(error)
            return np.array(radial_copy[0, 0]), np.array(total_counts), np.array(wo_bg_counts), np.array(
                background_avgs)

        elif band == "R":
            radial_copy[1, 0][mask1] *= 3
            radial_copy[1, 0][mask2] *= 5
            if err:
                return np.array(radial_copy[1, 0]), np.array(total_counts), np.array(wo_bg_counts), np.array(
                    background_avgs), np.array(error)
            return np.array(radial_copy[1, 0]), np.array(total_counts), np.array(wo_bg_counts), np.array(
                background_avgs)
        else:
            raise ValueError("ABORT wrong input")

    def mark_objects(self, inner_radius, outer_radius, err=False, alpha=0.5):
        img_i = self.images[0].data[0, :, :].copy()
        img_r = self.images[1].data[0, :, :].copy()

        shape = img_i.shape

        cmap = plt.cm.get_cmap('Set1_r')
        total_counts = []
        wo_bg_counts = []
        background_avgs = []
        error = []

        mask = np.zeros(shape)
        alphas = np.zeros(shape)

        for obj in self.objects:
            mask_in = aperture(shape, *obj.get_pos(False), inner_radius)
            mask_out = aperture(shape, *obj.get_pos(False), inner_radius + outer_radius, inner_radius)

            total_counts.append([np.sum(img_i[mask_in]), np.sum(img_r[mask_in])])
            background_avgs.append([np.mean(img_i[mask_out]), np.mean(img_r[mask_out])])
            wo_bg_counts.append([total_counts[-1][0] - background_avgs[-1][0] * np.sum(mask_in),
                                 total_counts[-1][1] - background_avgs[-1][1] * np.sum(mask_in)])

            mask += 0.5 * mask_in + mask_out
            alphas += alpha * (mask_in + mask_out)

            if err:
                error.append(np.sqrt(np.abs(total_counts[-1])))

        mask = cmap(mask)
        mask[..., -1] = alphas

        if err:
            return mask, np.array(total_counts), np.array(wo_bg_counts), np.array(background_avgs), np.array(error)

        return mask, np.array(total_counts), np.array(wo_bg_counts), np.array(background_avgs)

    def azimuthal_fitting(self, image: str):
        if image.lower() == "i":
            img = self.images[0].data[0].copy()
        elif image.lower() == "r":
            img = self.images[1].data[0].copy()
        else:
            raise ValueError("Wrong image name")
        profile = azimuthal_averaged_profile(img)

        # plt.figure()
        # plt.title(self.name + " Azimuthal profile and fit")
        # plt.semilogy(range(0, 512), profile)

        try:
            fitting = curve_fit(moffat_1d, *profile,
                                bounds=((0, 0, 0, 0, -np.inf), (5, np.inf, np.inf, np.inf, np.inf)))
            print("Azimuthal profil fitting paramter:")
            print(fitting[0])
            # plt.semilogy(range(0, 512), moffat_1d(np.arange(0, 512), *fitting[0]))
        except RuntimeError:
            print("The fitting didnt work :(")
            fitting = None

        return np.array(profile), fitting


def azimuthal_averaged_profile(image: np.ndarray, err=False):
    size = image[0].size
    shape = image.shape
    radius = size // 2
    img = image.copy()
    profile = []
    error = []
    for r in range(0, radius):
        mask = aperture(shape, size // 2, size // 2, r)

        profile.append(np.nanmean(img[mask]))
        n = len(img[mask])
        if n != 0:
            error.append(np.sqrt(np.nansum(img[mask])) / len(img[mask]))
        else:
            error.append(0)

        img[mask] = np.nan

    if err:
        return np.arange(0, radius), np.array(profile), np.array(error)

    return np.arange(0, radius), np.array(profile)


# todo fix this shit below
def half_azimuthal_averaged_profile(image: np.ndarray, err=False):
    size = image[0].size
    radius = size // 2
    cx, cy = size // 2, size // 2
    x, y = np.arange(0, 2 * radius), np.arange(0, 2 * radius)
    img = image.copy()
    profile = [img[radius, radius]]
    error = []
    for r in range(1, radius):
        neg_mask = ((x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 <= r ** 2) & \
                   (x[np.newaxis, :] < radius)
        pos_mask = ((x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 <= r ** 2) & \
                   (x[np.newaxis, :] >= radius)

        profile.append(np.average(img[pos_mask]))
        profile.insert(0, np.average(img[neg_mask]))
        error.append(np.std(img[pos_mask]) / np.sqrt(len(img[pos_mask])))
        error.insert(0, np.std(img[neg_mask]) / np.sqrt(len(img[neg_mask])))

    plt.show()
    if err:
        return np.arange(-radius, radius), np.array(profile), np.array(error)

    return np.arange(-radius, radius), np.array(profile)


def magnitude_wavelength_plot(fix_points, x):
    fit = np.polyfit(fix_points[:, 1], fix_points[:, 0], 1)
    p = np.poly1d(fit)
    plt.figure()
    plt.scatter(fix_points[:, 1], fix_points[:, 0], label="Star Fluxes")
    plt.scatter(x, p(x), label="Filter central wavelength", zorder=2)
    plt.plot([400, 2200], p([400, 2200]), c='green', label="fit: " + str(p), zorder=0)
    plt.legend()
    plt.xlabel("Wavelength in nm")
    plt.ylabel("Stellar Magnitude")
    print(p)
    print(p(x))
    print(p(x[1]) - p(x[0]))


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


def angle_phi(x, y, x0, y0):
    size = len(y)
    out = np.zeros((size, size))
    out[:, :x0] = -np.inf
    out[:, x0 + 1:] = np.inf
    results = np.true_divide(x - x0, y - y0, out=out, where=(x == x) & (y != y0))
    return np.arctan(results)


def aperture(shape, cx, cy, radius, hole=0):
    y, x = np.ogrid[:shape[0], :shape[1]]
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mask = (hole <= distance) & (distance < radius)
    return mask


def diffraction_rings(profile: np.ndarray, estimate: int, width: int = 6):
    size = len(profile)
    first_deriv = np.gradient(profile)
    second_deriv = np.gradient(first_deriv)
    estimates = np.arange(estimate - width, estimate + width)
    sec_deriv_func = interpolate.interp1d(np.arange(0, size), second_deriv)

    def sec_deriv_sq(x):
        return sec_deriv_func(x) ** 2

    zeros = least_squares(sec_deriv_sq, estimates, bounds=(1, estimate + width))
    return zeros.x, sec_deriv_sq(zeros.x)
