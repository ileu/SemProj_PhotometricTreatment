import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import curve_fit, root_scalar, RootResults
from scipy.fftpack import fft, ifft
import numpy as np
from astropy.io import fits
from StarFunctions import StarImg, OOI, azimuthal_averaged_profile, magnitude_wavelength_plot, diffraction_rings, \
    half_azimuthal_averaged_profile
import StarGUI
import DiskGUI
import gc

# todo: find sigma of ND4 background and estimate if visible; done, see ratio
# todo: try to recreate hd100453 peak with ND4 image; somewhat but ungh not happy
# todo: calculate flux of disk around hd100453; done

# todo: compare diffraction rings of the measurements as well as ghost and see if moved; kind of ->

# todo: look for disk in ND4; maybe but might be just imagination

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

""" function calls """

for obj in gc.get_objects():
    if isinstance(obj, StarImg):
        print(obj.name)
        x, radial = azimuthal_averaged_profile(obj.get_i_img()[0])
        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        ax1.set_title("azimuthal rofile")
        ax1.semilogy(x, radial)
        ax2 = fig.add_subplot(132)
        ax2.set_title("first derivative")
        ax2.plot(x, np.gradient(radial))
        ax3 = fig.add_subplot(133)
        ax3.set_title("second derivative")
        sec = np.gradient(np.gradient(radial))
        sec_deriv_func = interpolate.interp1d(x, sec)


        def sec_deriv_sq(x):
            return sec_deriv_func(x) ** 2


        thin = np.linspace(0, 500, 10000)
        ax3.plot(thin, sec_deriv_sq(thin))
        results = diffraction_rings(radial, 22)
        print(np.array2string(results[0], precision=2))
        print(np.array2string(results[1], precision=2))

        textax = plt.axes([0.5,0.95,0.3,0.03])
        textax.axis('off')
        textax.text(0, 0, obj.name, fontsize=18, ha='center')
        plt.show()

# StarGUI.start(ND4)

# DiskGUI.start(cyc116)

# cyc116_third_star.fitting_3d(cyc116.get_i_img()[0])

# plt.show()

# """  test 1 """
# radial1, fit1 = cyc116.azimuthal_fitting("i")
#
# radial2, fit2 = ND4.azimuthal_fitting("i")
# plt.show()
# """ Test 1 with ND4 """
# scaling_factor1 = np.average(radial1[1, 8:28] / radial2[1, 8:28])
# print("scaling factor ND4: ", scaling_factor1)
# scaling_func = lambda x, c: c * x
#
# scaling_factor2 = curve_fit(scaling_func, radial2[1], radial1[1])
# print("scaling factor 2", scaling_factor2)
#
# plt.figure()
# plt.title("Comparison 1 of ND4 to cyc116 manual")
# plt.semilogy(*radial1, label="profile of cyc116")
# plt.semilogy(radial2[0], scaling_factor1 * radial2[1], label="scaled profile of ND4")
# plt.semilogy([0, 150], [np.max(cyc116.get_i_img()[0]), np.max(cyc116.get_i_img()[0])], '--',
#              label="max value of cyc116")
# plt.legend()
# plt.ylim(ymin=1)
#
# plt.figure()
# plt.title("Comparison 1 of ND4 to cyc116 wtih fit")
# plt.semilogy(*radial1)
# plt.semilogy(radial2[0], scaling_factor2[0] * radial2[1])
# plt.semilogy([0, 150], [np.max(cyc116.get_i_img()[0]), np.max(cyc116.get_i_img()[0])], 'g--')
# plt.ylim(ymin=1)
#
# plt.show()
#
# """  test 2 """
# x, radial1 = half_azimuthal_averaged_profile(cyc116.get_i_img()[0])
# _, radial2 = half_azimuthal_averaged_profile(ND4.get_i_img()[0])
# plt.show()
#
# """ Test 1 with ND4 """
# scaling_factor2 = np.average(radial1[520:540] / radial2[520:540])
# scaling_factor1 = np.average(radial1[484:504] / radial2[484:504])
# print("scaling factor ND4: ", scaling_factor1)
# print("scaling factor ND4: ", scaling_factor2)
# scaling_func = lambda x, c: c * x
#
# plt.figure()
# plt.title("Comparison 2 of ND4 to cyc116")
# plt.semilogy(x, radial1, label="profile of cyc116")
# plt.semilogy(x, (scaling_factor1 + scaling_factor2) * 0.5 * radial2, label="scaled profile of ND4")
# plt.semilogy([-150, 150], [np.max(cyc116.get_i_img()[0]), np.max(cyc116.get_i_img()[0])], '--',
#              label="max value of cyc116")
# plt.ylim(ymin=1)
# plt.legend()

plt.show()
