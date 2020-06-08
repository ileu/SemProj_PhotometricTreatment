import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import curve_fit
import numpy as np
from astropy.io import fits
from StarFunctions import StarImg, OOI, azimuthal_averaged_profile, moffat_1d
from StarGUI import start, magnitude_wavelength_plot

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

# cyc116_third_star.fitting_3d(cyc116.get_i_img()[0])

# cyc116.azimuthal_fitting("i")

# start(cyc116)

# magnitude_wavelength_plot(HD100453_fluxes, [Iband_filter, Rband_filter])

# cyc116.calc_radial_polarization()

plt.show()

"""  testing star removal """
radial1, fit1 = cyc116.azimuthal_fitting("i")

radial2, fit2 = ND4.azimuthal_fitting("i")

radial3 = azimuthal_averaged_profile(cyc116_ghost1.local_map)
fit3 = curve_fit(moffat_1d, range(0, 40), radial3,
                 bounds=((0, 0, 0, 0, -np.inf), (5, np.inf, np.inf, np.inf, np.inf)))
plt.figure()
plt.title("Ghost 1 Azimuthal profile and fit")
plt.semilogy(range(0, 40), radial3)
plt.semilogy(range(0, 40), moffat_1d(range(0, 40), *fit3[0]))

radial4, fit4 = PointSpread.azimuthal_fitting("i")
plt.show()
""" Test 1 with ND4 """

scaling_factor2 = np.average(radial1[1, 9:18] / radial2[1, 9:18])
print("scaling factor ND4: ", scaling_factor2)

plt.figure()
plt.title("Comparison of ND4 to cyc116")
plt.semilogy(*radial1)
plt.semilogy(radial2[0], scaling_factor2 * moffat_1d(radial1[0], *fit2[0], offset=False))
plt.ylim(ymin=1)

plt.figure()
plt.title("Subtraction of main star ND4")
plt.semilogy(radial2[0], radial1[1] - scaling_factor2 * moffat_1d(radial1[0], *fit2[0], offset=False))

plt.show()

""" Test 2 with Ghost 1"""
scaling_factor3 = np.average(radial1[1, 9:18] / moffat_1d(radial1[0, 9:18], *fit3[0], offset=False))
print("scaling factor Ghost1: ", scaling_factor3)

plt.figure()
plt.title("Comparison of Ghost1 to cyc116")
plt.semilogy(*radial1)
plt.semilogy(radial2[0], scaling_factor3 * moffat_1d(radial1[0], *fit3[0], offset=False))
plt.ylim(ymin=1)

plt.figure()
plt.title("Subtraction of main star Ghost1")
plt.semilogy(radial2[0], radial1[1] - scaling_factor3 * moffat_1d(radial1[0], *fit3[0], offset=False))

plt.show()

""" Test 3 with PSF"""

scaling_factor4 = np.average(radial1[1, 9:18] / radial4[1, 9:18])
print("scaling factor PSF: ", scaling_factor4)

plt.figure()
plt.title("Comparison of Ghost1 to PSF")
plt.semilogy(*radial1)
plt.semilogy(radial2[0], scaling_factor4 * moffat_1d(radial1[0], *fit4[0], offset=False))
plt.ylim(ymin=1)

plt.figure()
plt.title("Subtraction of main star PSF")
plt.semilogy(radial2[0], radial1[1] - scaling_factor4 * moffat_1d(radial1[0], *fit4[0], offset=False))

plt.show()
