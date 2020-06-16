import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, root_scalar
from StarData import cyc116, ND4, PointSpread
from StarFunctions import azimuthal_averaged_profile, magnitude_wavelength_plot, diffraction_rings, \
    half_azimuthal_averaged_profile, annulus
import StarGUI
import DiskGUI

# StarGUI.start(HD163296)

# cyc116.calc_radial_polarization()

# DiskGUI.start(cyc116)

# plt.show()

first = 12
second = 45
y_min = 0.1
tail = np.linspace(120, 200, 8, dtype=int, endpoint=False)
nd4_region = np.concatenate((np.arange(first, second), tail))
psf_region = np.concatenate((np.arange(20), tail))
markers_on = [first, second, *tail]
markers_on2 = [20, *tail]

profile = ["I-band", "R-band"]


def scaling_func(pos, a, b):
    return a * (pos - b)


def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    adjust_yaxis(ax2, (y1 - y2) / 2, v2)
    adjust_yaxis(ax1, (y2 - y1) / 2, v1)


def adjust_yaxis(ax, ydif, v):
    """shift axis ax by ydiff, maintaining point v at the same location"""
    inv = ax.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, ydif))
    miny, maxy = ax.get_ylim()
    miny, maxy = miny - v, maxy - v
    if -miny > maxy or (-miny == maxy and dy > 0):
        nminy = miny
        nmaxy = miny * (maxy + dy) / (miny + dy)
    else:
        nmaxy = maxy
        nminy = maxy * (miny + dy) / (maxy + dy)
    ax.set_ylim(nminy + v, nmaxy + v)


for index in [0, 1]:
    print(profile[index])
    x, radial1 = cyc116.azimuthal[index]
    _, radial2 = ND4.azimuthal[index]

    _, psf = PointSpread.azimuthal[index]

    x2, qphi = cyc116.azimuthal_qphi[index]

    guess = (1.0 / ND4.filter_reduction[index], np.median(radial2[100:]))
    print("guess: ", guess)

    scaling_factor = curve_fit(scaling_func, radial2[nd4_region], radial1[nd4_region], p0=guess,
                               sigma=radial1[nd4_region])
    scaled_profile = scaling_func(radial2, *scaling_factor[0])

    psf_factor = curve_fit(scaling_func, psf[psf_region], scaled_profile[psf_region], p0=guess,
                           sigma=scaled_profile[psf_region])

    psf_factor1 = scaling_func(radial2[0], *scaling_factor[0]) / psf[0]

    psf_profile = scaling_func(psf, *psf_factor[0])

    print("scaling factor", scaling_factor)
    print("psf factor", psf_factor)
    print("psf factor1", psf_factor1)

    fig = plt.figure(figsize=(28, 14))
    textax = plt.axes([0.5, 0.95, 0.3, 0.03])
    textax.axis('off')
    textax.text(0, 0, "Comparison " + profile[index] + " of ND4 to cyc116", fontsize=18, ha='center')

    # ax = fig.add_subplot(1, 2, 1)
    # ax.set_title("manual")
    # ax.semilogy(x, radial1, '-D', label="profile of cyc116", markevery=markers_on)
    # ax.semilogy(x, guess[0] * (radial2 - guess[1]), label="scaled profile of ND4")
    #
    # ax.semilogy(x, psf_factor1 * psf, label="PSF profile")
    # ax.legend()
    # ax.set_ylim(ymin=y_min)

    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy(x, radial1, '-D', label="profile of cyc116", markevery=markers_on)
    ax.semilogy(x, scaled_profile, label="scaled profile of ND4")

    ax.semilogy(x, psf_profile, '-D', label="PSF profile", markevery=markers_on2)
    ax.legend(fontsize='large')
    ax.set_ylim(ymin=y_min)

    axins = ax.inset_axes([0.35, 0.55, 0.5, 0.43])
    axins.semilogy(x, radial1, '-D', label="profile of cyc116", markevery=markers_on)
    axins.semilogy(x, scaled_profile, label="scaled profile of ND4")
    axins.semilogy(x, psf_profile, '-D', label="PSF profile", markevery=markers_on2)
    axins.set_ylim((0.9 * np.min(psf_profile[:110]), 1.2 * 1e6))
    axins.set_xlim((-2, 110))
    ax.indicate_inset_zoom(axins)

    fig.savefig("../Bilder/Comparison_" + profile[index] + ".png", dpi=150)

    fig = plt.figure(figsize=(32, 14))
    textax = plt.axes([0.5, 0.95, 0.3, 0.03])
    textax.axis('off')
    textax.text(0, 0, "Subtraction in " + profile[index], fontsize=18, ha='center')

    # ax = fig.add_subplot(1, 3, 1)
    # ax.set_title("manual")
    # ax.plot(x, radial1 - psf_factor1 * psf, )
    # ax.set_ylim(ymin=-100, ymax=1.1 * max(radial1[20:] - psf_factor1 * psf[20:]))

    ax = fig.add_subplot(1, 1, 1)
    line1, = ax.plot(x, radial1 - psf_profile, label="Reduced cyc116 profile")
    ax1 = ax.twinx()
    line2, = ax1.plot(x2[20:], qphi[20:], "C3", label="Qphi profile")
    ax.tick_params(axis='y', labelcolor="C0")
    ax1.tick_params(axis='y', labelcolor="C3")
    ax1.set_ylim(ymin=-10, ymax=1.1 * max(qphi[20:]))
    ax.set_ylim(ymin=-100, ymax=1.1 * max(radial1[20:] - psf_profile[20:]))
    line3 = ax.axhline(0, ls='--', c='k', alpha=0.5, label="zero")
    lines = [line1, line2, line3]
    align_yaxis(ax, 0, ax1, 0)
    ax.legend(lines, [line.get_label() for line in lines],fontsize='large')

    # ax = fig.add_subplot(1, 3, 3)
    # ax.set_title("Qphi profile")
    # ax.plot(x2, qphi)
    # ax.set_ylim(ymin=-100, ymax=1.1 * max(qphi[20:]))
    # plt.show()
    #
    fig.savefig("../Bilder/Subtraction_" + profile[index] + ".png", dpi=150)

    disk_profile = radial1 - psf_profile
    disk_profile[disk_profile < 0] = 0

    print("Counts fit: ", np.sum(disk_profile[32:125]))

""" 2d attempt """

# first = 11
# second = 38
# y_min = 0.1
# markers_on = [first, second]
# profile = ["I-band", "R-band"]
# a = 1000
#
# for index in [0]:
#     print(profile[index])
#     map1 = cyc116.get_i_img()[index]
#     map2 = ND4.get_i_img()[index]
#     _, radial2 = ND4.azimuthal[index]
#     psf = PointSpread.get_i_img()[index]
#
#     qphi = cyc116.radial[index, 0]
#
#     region = annulus(map1, second, first)
#     outer_region = annulus(map1, np.inf, 200)
#     disk = annulus(map1, 125, 32)
#
#     guess = (1.0 / ND4.filter_reduction[index], np.median(radial2[100:]))
#     print("guess: ", guess)
#     max_cor = np.argmax(map2[region])
#     print("maximum location", max_cor)
#     print(map2[region].flatten()[max_cor])
#     psf_factor = guess[0] * (map2[region].flatten()[max_cor] - guess[1]) / psf[region].flatten()[max_cor]
#     print("psf factor:", psf_factor)

# plt.figure()
# plt.semilogy(range(0, 1024), map1[512, :])
# plt.semilogy(range(0, 1024), guess[0] * (map2[512, :] - guess[1]))  # links rechts
# plt.semilogy(range(0, 1024), psf_factor * psf[512, :])
#
# plt.figure()
# plt.semilogy(range(0, 1024), map1[:, 512])
# plt.semilogy(range(0, 1024), guess[0] * (map2[:, 512] - guess[1]))  # oben unten
# plt.semilogy(range(0, 1024), psf_factor * psf[:, 512])
#
# plt.figure()
# plt.semilogy(range(0, 1024), map1[512, :] - psf_factor * psf[512, :])
#
# plt.figure()
# plt.semilogy(range(0, 1024), map1[:, 512] - psf_factor * psf[:, 512])

# disk_map = map1 - psf_factor * psf
# disk_map[disk_map < 0] = 0
# print("Counts manual: ", np.sum(disk_map[disk]))

# for obj in gc.get_objects():
#     if isinstance(obj, StarImg):
#         print(obj.name)
#         x, radial = azimuthal_averaged_profile(obj.get_i_img()[0])
#         # x, radial = half_azimuthal_averaged_profile(obj.get_i_img()[0])
#         fig = plt.figure(num=obj.name)
#         ax1 = fig.add_subplot(131)
#         ax1.set_title("azimuthal rofile")
#         ax1.semilogy(x, radial)
#         ax2 = fig.add_subplot(132)
#         ax2.set_title("first derivative")
#         ax2.plot(x, np.gradient(radial))
#         ax3 = fig.add_subplot(133)
#         ax3.set_title("second derivative")
#         sec = np.gradient(np.gradient(radial))
#         sec_deriv_func = interpolate.interp1d(x, sec)
#
#
#         def sec_deriv_sq(x):
#             return sec_deriv_func(x) ** 2
#
#
#         thin = np.linspace(x[0], x[-1], 10000)
#         ax3.plot(thin, sec_deriv_sq(thin))
#         results = diffraction_rings(radial, 19, width=10)
#         print(np.array2string(results[0], precision=2))
#         print(np.array2string(results[1], precision=2))
#
#         textax = plt.axes([0.5, 0.95, 0.3, 0.03])
#         textax.axis('off')
#         textax.text(0, 0, obj.name, fontsize=18, ha='center')

plt.show()
