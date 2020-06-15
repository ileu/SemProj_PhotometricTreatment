import matplotlib.pyplot as plt
import mayavi.mlab as mlab
from scipy import interpolate
from scipy.signal import medfilt
from scipy.optimize import curve_fit, root_scalar
from scipy.fft import fft, ifft
import numpy as np
from astropy.io import fits
from StarFunctions import StarImg, OOI, azimuthal_averaged_profile, magnitude_wavelength_plot, diffraction_rings, \
    half_azimuthal_averaged_profile, annulus
import StarGUI
import DiskGUI
import gc

# airy function first zero at 3.8317; second zero at 7.0156; third zero at 13.3237
# or bessel1/x  first max at 0      ; second max at 5.13568; third max at 8.841724

""" import data """

cyc116_1_data = fits.open("../Data/sci_cyc116_1.fits")
cyc116_2_data = fits.open("../Data/sci_cyc116_2.fits")

cyc116 = StarImg("cyc116", cyc116_1_data[0], cyc116_2_data[0])

ND4_1_data = fits.open("../Data/sci_ND4_1.fits")
ND4_2_data = fits.open("../Data/sci_ND4_2.fits")

ND4 = StarImg("ND4", ND4_1_data[0], ND4_2_data[0])

ND4_filter_data = np.loadtxt("../Data/ND4_filter.txt", delimiter="\t", skiprows=1)

HD142527_1_data = fits.open("../Data/sci_HD142527_norm_1.fits")
HD142527_2_data = fits.open("../Data/sci_HD142527_norm_2.fits")

HD142527 = StarImg("HD142527", HD142527_1_data[0], HD142527_2_data[0])

HD163296_1_data = fits.open("../Data/sci_HD163296_R_I_fpnorm_1.fits")
HD163296_2_data = fits.open("../Data/sci_HD163296_R_I_fpnorm_2.fits")

HD163296 = StarImg("HD163296", HD163296_1_data[0], HD163296_2_data[0])

HD169142_1_data = fits.open("../Data/sci_HD169142_Inorm_1.fits")
HD169142_2_data = fits.open("../Data/sci_HD169142_Rnorm_2.fits")

HD169142 = StarImg("HD169142", HD169142_1_data[0], HD169142_2_data[0])

Spread_1_data = fits.open("../Data/PSFiband.fits")
Spread_2_data = fits.open("../Data/PSFrband.fits")

PointSpread = StarImg("Point Spread", Spread_1_data[0], Spread_2_data[0])

""" designate objects """

""" central wavelength of filter """
Rband_filter = 636.3
Iband_filter = 789.7

""" filter magnitudes B; V; G; J; H; K; """
HD100453_fluxes = np.array([[8.09, 445], [7.79, 551], [7.7196, 464], [6.945, 1220], [6.39, 1630], [5.6, 2190]])
HD142527_fluxes = np.array([[9.04, 445], [8.34, 551], [8.0940, 464], [6.503, 1220], [5.715, 1630], [4.980, 2190]])
HD163296_fluxes = np.array([[6.93, 445], [6.85, 551], [6.8111, 464], [6.195, 1220], [5.31, 1630], [4.779, 2190]])
# /\ additional U: 7.00 365nm R: 6.86 658nm I: 6.67 806nm
HD169142_fluxes = np.array([[8.42, 445], [8.16, 551], [8.0545, 464], [7.310, 1220], [6.911, 1630], [6.41, 2190]])

ND4_filter = interpolate.interp1d(ND4_filter_data[:, 0], ND4_filter_data[:, 3])

cyc116.set_filter(1, 1)
ND4.set_filter(ND4_filter(Iband_filter), ND4_filter(Rband_filter))
HD142527.set_filter(1, 1)
HD163296.set_filter(1, 1)
HD169142.set_filter(1, 1)

cyc116_second_star = OOI("Second star", 301, 307)
cyc116_third_star = OOI("Third star", 298, 724)
cyc116_ghost1 = OOI("Ghost 1", 237, 386)
cyc116_ghost2 = OOI("Ghost 2", 891, 598)
cyc116_main_star = OOI("Main Star", 511, 512)
cyc116_disk = OOI("Disk", 509, 509)

ND4_second_star = OOI("Second star", 301, 307)
ND4_third_star = OOI("Third star", 298, 724)
ND4_ghost1 = OOI("Ghost 1", 237, 386)
ND4_ghost2 = OOI("Ghost 2", 891, 598)
ND4_main_star = OOI("Main Star", 512, 512)

cyc116.add_object(cyc116_second_star)
cyc116.add_object(cyc116_third_star)
cyc116.add_object(cyc116_ghost1)
cyc116.add_object(cyc116_ghost2)
cyc116.add_object(cyc116_main_star)

cyc116.set_disk(cyc116_disk)

# ND4.add_object(ND4_second_star)
# ND4.add_object(ND4_third_star)
# ND4.add_object(ND4_ghost1)
# ND4.add_object(ND4_ghost2)
ND4.add_object(ND4_main_star)

HD142527_ghost1 = OOI("Ghost 1", 216, 386)
HD142527_ghost2 = OOI("Ghost 2", 894, 599)
HD142527_main_star = OOI("Main Star", 512, 512)

HD142527.add_object(HD142527_ghost1)
HD142527.add_object(HD142527_ghost2)
HD142527.add_object(HD142527_main_star)

HD163296_ghost1 = OOI("Ghost 1", 601, 515)
HD163296_ghost2 = OOI("Ghost 2", 428, 506)
HD163296_main_star = OOI("Main Star", 512, 512)

HD163296.add_object(HD163296_ghost1)
HD163296.add_object(HD163296_ghost2)
HD163296.add_object(HD163296_main_star)

HD169142_main_star = OOI("Main Star", 512, 512)

HD169142.add_object(HD169142_main_star)

PointSpread_main_star = OOI("Main Star", 512, 512)

PointSpread.add_object(PointSpread_main_star)
""" save and calculate """

# for obj in gc.get_objects():
#     if isinstance(obj, StarImg):
#         print(obj.name)
#         obj.save()

""" load data """
for obj in gc.get_objects():
    if isinstance(obj, StarImg):
        obj.load()

""" function calls """

# StarGUI.start(HD163296)

# cyc116.calc_radial_polarization()

# DiskGUI.start(cyc116)

# cyc116_third_star.fitting_3d(cyc116.get_i_img()[0])

# plt.show()

first = 12
second = 45
y_min = 0.1
region = np.concatenate((np.arange(first, second), np.linspace(100, 200, 5, dtype=int)))
print(region)
markers_on = [first, second]
profile = ["I-band", "R-band"]


def scaling_func(pos, a, b):
    return a * (pos - b)


for index in [0, 1]:
    print(profile[index])
    x, radial1 = cyc116.azimuthal[index]
    _, radial2 = ND4.azimuthal[index]

    _, psf = PointSpread.azimuthal[index]

    x2, qphi = cyc116.azimuthal_qphi[index]

    guess = (1.0 / ND4.filter_reduction[index], np.median(radial2[100:]))
    print("guess: ", guess)

    scaling_factor = curve_fit(scaling_func, radial2[region], radial1[region], p0=guess, sigma=radial1[region])

    psf_factor1 = guess[0] * (radial2[0] - guess[1]) / psf[0]

    psf_factor2 = scaling_func(radial2[0], *scaling_factor[0]) / psf[0]

    print("scaling factor", scaling_factor)

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
    ax.set_title("fit")
    ax.semilogy(x, radial1, '-D', label="profile of cyc116", markevery=markers_on)
    ax.semilogy(x, scaling_func(radial2, *scaling_factor[0]), label="scaled profile of ND4")

    ax.semilogy(x, psf_factor2 * psf, label="PSF profile")
    ax.legend()
    ax.set_ylim(ymin=y_min)

    fig.savefig("../Bilder/Comparison_" + profile[index] + ".png", dpi=300)

    fig = plt.figure(figsize=(32, 14))
    textax = plt.axes([0.5, 0.95, 0.3, 0.03])
    textax.axis('off')
    textax.text(0, 0, "Subtraction in " + profile[index], fontsize=18, ha='center')

    # ax = fig.add_subplot(1, 3, 1)
    # ax.set_title("manual")
    # ax.plot(x, radial1 - psf_factor1 * psf, )
    # ax.set_ylim(ymin=-100, ymax=1.1 * max(radial1[20:] - psf_factor1 * psf[20:]))

    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("fit")
    line1, = ax.plot(x, radial1 - psf_factor2 * psf, label="Reduced cyc116 profile")
    ax1 = ax.twinx()
    line2, = ax1.plot(x2, qphi, "C3", label="Qphi profile")
    ax.tick_params(axis='y', labelcolor="C0")
    ax1.tick_params(axis='y', labelcolor="C3")
    ax1.set_ylim(ymin=-10, ymax=1.1 * max(qphi[20:]))
    ax.set_ylim(ymin=-100, ymax=1.1 * max(radial1[20:] - psf_factor2 * psf[20:]))
    lines = [line1, line2]
    ax.legend(lines, [l.get_label() for l in lines])

    # ax = fig.add_subplot(1, 3, 3)
    # ax.set_title("Qphi profile")
    # ax.plot(x2, qphi)
    # ax.set_ylim(ymin=-100, ymax=1.1 * max(qphi[20:]))
    # plt.show()
    #
    fig.savefig("../Bilder/Subtraction_" + profile[index] + ".png", dpi=300)

    disk_profile1 = radial1 - psf_factor1 * psf
    disk_profile2 = radial1 - psf_factor2 * psf
    disk_profile1[disk_profile1 < 0] = 0
    disk_profile2[disk_profile2 < 0] = 0

    print("Counts manual: ", np.sum(disk_profile1[32:125]))
    print("Counts fit: ", np.sum(disk_profile2[32:125]))

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
