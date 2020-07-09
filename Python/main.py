import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import sigmaclip
from StarData import cyc116, ND4, PointSpread, cyc116_second_star, cyc116_ghost2, ND4_filter_data, HD100453_fluxes, \
    Rband_filter, Iband_filter
from datetime import datetime
from scipy.ndimage import gaussian_filter1d
import StarGUI
import DiskGUI
from StarFunctions import aperture, magnitude_wavelength_plot, photometrie_poly, photometrie, photometrie_disk


def scaling_func(pos, a, b):
    return a * (pos - b)


def scaling_gauss_func(pos, a, b, sig):
    return a * (gaussian_filter1d(pos, sig) - b)


def mkdir_p(mypath):
    """Creates a directory. equivalent to using mkdir -p on the command line"""

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise


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


def annulus_plot():
    size = 150
    x, y = np.meshgrid(range(0, size), range(0, size))
    distance = np.sqrt((x - size // 2) ** 2 + (y - size // 2) ** 2)
    mask1 = np.where((0 <= distance) & (distance < 25), 0.5, 1)
    mask2 = np.where((25 <= distance) & (distance < 50), 0, 1)
    fig_an = plt.figure()
    ax_an = fig_an.add_subplot()
    ax_an.tick_params(labelsize=14)
    ax_an.locator_params(axis='both', nbins=8)
    ax_an.imshow(mask1 + mask2, cmap='Set1')
    fig_an.savefig("../Bilder/Annulus.png", dpi=150, bbox_inches='tight', pad_inches=0.1)


def filter_plot():
    fig_filter = plt.figure(figsize=(12, 7))
    ax_filter = fig_filter.add_subplot()
    ax_filter.semilogy(ND4_filter_data[:, 0], ND4_filter_data[:, 3], label="ND4")
    ax_filter.set_xlabel("wavelength [nm]", fontsize=16)
    ax_filter.set_ylabel("transmission", fontsize=16)
    # ax_filter.set_yticks([1e-3, 1e-4, 1e-5])
    ax_filter.tick_params(labelsize=14)
    ax_filter.legend(fontsize="large")
    ax_filter.grid(which="both", alpha=0.25, zorder=-2, c='k')
    fig_filter.savefig("../Bilder/nd4_filter.png", dpi=150, bbox_inches='tight', pad_inches=0.1)


def overview_plot():
    plt.figure(figsize=(8, 8))
    plt.imshow(cyc116.radial[0, 0], vmin=-50, vmax=110, cmap='gray')
    plt.ylim((362, 662))
    plt.xlim((362, 662))
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)


def disk_plot():
    plt.figure(figsize=(8, 8))
    plt.imshow(np.log10(1e-2 * cyc116.get_i_img()[0] + 1), cmap='gray')
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)


def aperture_photometrie():
    print("------- Aperture -------")
    print("Big aperture")
    results_big = []
    for observation in [cyc116, ND4, PointSpread]:
        print(observation.name)
        result = photometrie(416, 466, (512, 512), observation.get_i_img(), observation.get_r_img())
        results_big.append(result)
        print(result)
        print(result[1] / result[0])
        print()

    # print("rescaled PSF")
    #
    # data_iq = scal_profiles[0] * circumference
    # data_iu = scal_profiles[1] * circumference
    # data_rq = scal_profiles[2] * circumference
    # data_ru = scal_profiles[3] * circumference
    # res_scal_iq = []
    # res_scal_iu = []
    # res_scal_rq = []
    # res_scal_ru = []
    # res_scal_small_iq = []
    # res_scal_small_iu = []
    # res_scal_small_rq = []
    # res_scal_small_ru = []
    # for rad_displ in np.arange(-1, 2):
    #     res_scal_iq.append(np.sum(data_iq[:(416 + rad_displ)]) - np.sum(circumference[:(416 + rad_displ)]) *
    #                        np.median(sigmaclip(scal_profiles[0][(416 + rad_displ):467])[0]))
    #     res_scal_iu.append(np.sum(data_iu[:(416 + rad_displ)]) - np.sum(circumference[:(416 + rad_displ)]) *
    #                        np.median(sigmaclip(scal_profiles[1][(416 + rad_displ):467])[0]))
    #     res_scal_rq.append(np.sum(data_rq[:(416 + rad_displ)]) - np.sum(circumference[:(416 + rad_displ)]) *
    #                        np.median(sigmaclip(scal_profiles[2][(416 + rad_displ):467])[0]))
    #     res_scal_ru.append(np.sum(data_ru[:(416 + rad_displ)]) - np.sum(circumference[:(416 + rad_displ)]) *
    #                        np.median(sigmaclip(scal_profiles[3][(416 + rad_displ):467])[0]))
    #
    #     res_scal_small_iq.append(np.sum(data_iq[:(20 + rad_displ)]) - np.sum(circumference[:(20 + rad_displ)]) *
    #                              np.median(sigmaclip(scal_profiles[0][(20 + rad_displ):40])[0]))
    #     res_scal_small_iu.append(np.sum(data_iu[:(20 + rad_displ)]) - np.sum(circumference[:(20 + rad_displ)]) *
    #                              np.median(sigmaclip(scal_profiles[1][(20 + rad_displ):40])[0]))
    #     res_scal_small_rq.append(np.sum(data_rq[:(20 + rad_displ)]) - np.sum(circumference[:(20 + rad_displ)]) *
    #                              np.median(sigmaclip(scal_profiles[2][(20 + rad_displ):40])[0]))
    #     res_scal_small_ru.append(np.sum(data_ru[:(20 + rad_displ)]) - np.sum(circumference[:(20 + rad_displ)]) *
    #                              np.median(sigmaclip(scal_profiles[3][(20 + rad_displ):40])[0]))
    #
    # print([np.mean(res_scal_iq), np.mean(res_scal_rq)], [np.std(res_scal_iq) / np.mean(res_scal_iq),
    #                                                      np.std(res_scal_rq) / np.mean(res_scal_rq)])
    # print([np.mean(res_scal_iu), np.mean(res_scal_ru)], [np.std(res_scal_iu) / np.mean(res_scal_iu),
    #                                                      np.std(res_scal_ru) / np.mean(res_scal_ru)])
    # results_big_scaled = np.array([np.mean(res_scal_iq), np.mean(res_scal_rq),
    #                                np.mean(res_scal_iu), np.mean(res_scal_ru)])
    # print()

    print("Mixed profile")

    data_iq = mixed_profiles[0] * circumference
    data_iu = mixed_profiles[1] * circumference
    data_rq = mixed_profiles[2] * circumference
    data_ru = mixed_profiles[3] * circumference
    res_iq = []
    res_iu = []
    res_rq = []
    res_ru = []
    res_small_iq = []
    res_small_iu = []
    res_small_rq = []
    res_small_ru = []
    for rad_displ in np.arange(-1, 2):
        res_iq.append(np.sum(data_iq[:(416 + rad_displ)]) - np.sum(circumference[:(416 + rad_displ)]) *
                      np.median(sigmaclip(mixed_profiles[0][(416 + rad_displ):467])[0]))
        res_iu.append(np.sum(data_iu[:(416 + rad_displ)]) - np.sum(circumference[:(416 + rad_displ)]) *
                      np.median(sigmaclip(mixed_profiles[1][(416 + rad_displ):467])[0]))
        res_rq.append(np.sum(data_rq[:(416 + rad_displ)]) - np.sum(circumference[:(416 + rad_displ)]) *
                      np.median(sigmaclip(mixed_profiles[2][(416 + rad_displ):467])[0]))
        res_ru.append(np.sum(data_ru[:(416 + rad_displ)]) - np.sum(circumference[:(416 + rad_displ)]) *
                      np.median(sigmaclip(mixed_profiles[3][(416 + rad_displ):467])[0]))

        res_small_iq.append(np.sum(data_iq[:(20 + rad_displ)]) - np.sum(circumference[:(20 + rad_displ)]) *
                            np.median(sigmaclip(mixed_profiles[0][(20 + rad_displ):40])[0]))
        res_small_iu.append(np.sum(data_iu[:(20 + rad_displ)]) - np.sum(circumference[:(20 + rad_displ)]) *
                            np.median(sigmaclip(mixed_profiles[1][(20 + rad_displ):40])[0]))
        res_small_rq.append(np.sum(data_rq[:(20 + rad_displ)]) - np.sum(circumference[:(20 + rad_displ)]) *
                            np.median(sigmaclip(mixed_profiles[2][(20 + rad_displ):40])[0]))
        res_small_ru.append(np.sum(data_ru[:(20 + rad_displ)]) - np.sum(circumference[:(20 + rad_displ)]) *
                            np.median(sigmaclip(mixed_profiles[3][(20 + rad_displ):40])[0]))

    print([np.mean(res_iq), np.mean(res_rq)], [np.std(res_iq) / np.mean(res_iq), np.std(res_rq) / np.mean(res_rq)])
    print([np.mean(res_iu), np.mean(res_ru)], [np.std(res_iu) / np.mean(res_iu), np.std(res_ru) / np.mean(res_ru)])
    results_big_mixed = np.array([np.mean(res_iq), np.mean(res_iu), np.mean(res_rq), np.mean(res_ru)])
    print()

    magnitude_wavelength_plot(HD100453_fluxes, (Rband_filter, Iband_filter))

    print("small aperture")
    results_small_cyc = []
    results_small_nd4 = []
    results_small_psf = []
    print("cyc116")
    print()
    for obj in cyc116.get_objects():
        results_small_cyc.append(photometrie(20, 39, obj.get_pos(), cyc116.get_i_img(), cyc116.get_r_img()))
        print(obj.name)
        print(results_small_cyc[-1])
        print(results_small_cyc[-1][1] / results_small_cyc[-1][0])
        print()

    print("ND4")
    print()
    for obj in ND4.get_objects():
        results_small_nd4.append(photometrie(20, 39, obj.get_pos(), ND4.get_i_img(), ND4.get_r_img()))
        print(obj.name)
        print(results_small_nd4[-1])
        print(results_small_nd4[-1][1] / results_small_nd4[-1][0])
        print()

    print("PSF")
    print()
    for obj in PointSpread.get_objects():
        results_small_psf.append(photometrie(20, 39, obj.get_pos(), PointSpread.get_i_img(), PointSpread.get_r_img()))
        print(obj.name)
        print(results_small_psf[-1])
        print(results_small_psf[-1][1] / results_small_psf[-1][0])
        print()

    print("Mixed")
    print()
    print([np.mean(res_small_iq), np.mean(res_small_iu), np.mean(res_small_rq), np.mean(res_small_ru)],
          [np.std(res_small_iq) / np.mean(res_small_iq), np.std(res_small_iu) / np.mean(res_small_iu),
           np.std(res_small_rq) / np.mean(res_small_rq), np.std(res_small_ru) / np.mean(res_small_ru)])

    results_small_mixed = np.array([np.mean(res_small_iq), np.mean(res_small_iu), np.mean(res_small_rq),
                                    np.mean(res_small_ru)])
    print()

    # print("Scaled")
    # print()
    # print([np.mean(res_scal_small_iq), np.mean(res_scal_small_iu),
    #        np.mean(res_scal_small_rq), np.mean(res_scal_small_ru)],
    #       [np.std(res_scal_small_iq) / np.mean(res_scal_small_iq),
    #        np.std(res_scal_small_iu) / np.mean(res_scal_small_iu),
    #        np.std(res_scal_small_rq) / np.mean(res_scal_small_rq),
    #        np.std(res_scal_small_ru) / np.mean(res_scal_small_ru)])
    #
    # results_small_scaled = np.array([np.mean(res_scal_small_iq), np.mean(res_scal_small_iu), np.mean(res_scal_small_rq),
    #                                  np.mean(res_scal_small_ru)])
    print()

    print("Disk")
    print()
    results_disk = photometrie_disk(28, 93, 124, cyc116.disk.get_pos(), cyc116.radial[0][0], cyc116.radial[1][0])
    print(results_disk)
    print(results_disk[1] / results_disk[0])
    print()

    print("Q frame")
    results_u = photometrie_disk(28, 93, 124, cyc116.disk.get_pos(), cyc116.get_i_img()[1], cyc116.get_r_img()[1],
                                 bg=True)
    print(results_u)
    print(results_u[1] / results_u[0])
    print()

    print("U frame")
    results_u = photometrie_disk(28, 93, 124, cyc116.disk.get_pos(), cyc116.get_i_img()[3], cyc116.get_r_img()[3],
                                 bg=True)
    print(results_u)
    print(results_u[1] / results_u[0])
    print()

    print("----- 3d Background -----")
    radius_range = np.arange(-3, 4)

    results_3d_sec = []
    results_3d_g2 = []
    for inner_range in radius_range:
        results_3d_sec.append(photometrie_poly(20, 39 + inner_range, cyc116_second_star.get_pos(),
                                               cyc116.get_i_img()[0]))
        results_3d_g2.append(photometrie_poly(20, 39 + inner_range, cyc116_ghost2.get_pos(), cyc116.get_i_img()[0]))

    results_sec = photometrie(20, 39, cyc116_second_star.get_pos(), cyc116.get_i_img(), cyc116.get_r_img(), displ=0,
                              scale=3)
    results_g2 = photometrie(20, 39, cyc116_ghost2.get_pos(), cyc116.get_i_img(), cyc116.get_r_img(), displ=0, scale=3)

    print("Companion")
    print(np.mean(results_3d_sec), np.std(results_3d_sec))
    print(results_sec)
    print("Ghost 2")
    print(np.mean(results_3d_g2), np.std(results_3d_g2))
    print(results_g2)
    print("IU")
    photometrie_poly(20, 39, cyc116_second_star.get_pos(), cyc116.get_i_img()[2])
    photometrie_poly(20, 39, cyc116_ghost2.get_pos(), cyc116.get_i_img()[2])
    print()

    print("----- Ratios -----")
    print("Big vs small")
    print()
    print("PSF")
    print(np.array(results_small_psf[0][0]) / np.array(results_big[2][0]))
    print(-2.5 * np.log10(np.array(results_small_psf[0][0]) / np.array(results_big[2][0])))
    print()
    print("Mixed")
    print(np.array(results_small_mixed) / np.array(results_big_mixed))
    print(-2.5 * np.log10(np.array(results_small_mixed) / np.array(results_big_mixed)))
    print()
    print("cyc116")
    print()
    for ind, obj in enumerate(cyc116.get_objects()):
        print(obj.name)
        small_ratio = results_small_cyc[ind][0] / np.array(results_small_mixed)
        print(small_ratio)
        print(np.mean(small_ratio[:2]), np.mean(small_ratio[2:]))
        print(-2.5 * np.log10(np.mean(small_ratio[:2])) + 7.42, -2.5 * np.log10(np.mean(small_ratio[2:])) + 7.6)
        print()
    # print("Scaled")
    # print(np.array(results_small_scaled) / np.array(results_big_scaled))
    # print()


""" GUI """

# DiskGUI.start(cyc116)
#
# StarGUI.start(cyc116)
#
# StarGUI.start(ND4)

""" Plots """

# annulus_plot()
#
filter_plot()
#
# overview_plot()
#
# disk_plot()

""" Fits """
print("--------- Fitting ---------")
start_int = 14
end_int = 65
start_peak = 0
end_peak = 32
transition = 21
y_min = 0.1
tail = np.linspace(120, 200, 8, dtype=int, endpoint=False)

nd4_region = np.arange(start_int, end_int)
psf_region = np.concatenate((np.arange(start_peak, end_peak), tail))

markers_on_nd4 = [start_int, end_int]
markers_on_psf = [end_peak, *tail]
weights_nd4 = np.concatenate((np.full((21 - start_int,), 1), np.full((end_int - 21,), 1)))
weights_psf = np.concatenate((np.full((end_peak - start_peak,), 15), np.full_like(tail, 1)))
bounds_psf = ([0, -np.inf, 0], np.inf)

save = True
smart = False

results = []
scal_profiles = []
mixed_profiles = []
star_profiles = []

circumference = [np.sum(aperture((1024, 1024), 512, 512, r, r - 1)) for r in range(1, 513)]
profile = ["I-band", "I-band $I_U$", "R-band", "R-band $I_U$"]

if save:
    folder = datetime.now().strftime('%d_%m_%H%M')
    path = "../Bilder/" + folder
    mkdir_p(path)

    param_file = open(path + "/parameters.txt", "w")

for index, _ in enumerate(profile):
    print(profile[index])
    radi, cyc116_profile = cyc116.azimuthal[index]
    _, nd4_profile = ND4.azimuthal[index]

    _, psf_profile = PointSpread.azimuthal[index]

    x2, qphi = cyc116.azimuthal_qphi[index // 2]

    guess = (1.0 / ND4.filter_reduction[index // 2], np.median(nd4_profile[100:]))
    print("guess: ", guess)

    scaling_factor = curve_fit(scaling_func, nd4_profile[nd4_region], cyc116_profile[nd4_region], p0=guess,
                               sigma=weights_nd4)
    print("scaling factor", scaling_factor)

    scaled_profile = scaling_func(nd4_profile, *scaling_factor[0])
    scal_profiles.append(scaled_profile)

    mixed_profile = cyc116_profile.copy()
    mixed_profile[:transition] = scaled_profile[:transition]
    mixed_profiles.append(mixed_profile)

    if smart:
        tail = np.array((np.abs(scaled_profile[120:200] - cyc116_profile[120:200]) <= 0.6827).nonzero()) \
               + 120
        print("Points satisfy condition: ", len(tail[0]))
        tail = tail[0, ::-len(tail[0]) // 8]
        tail = tail[::-1]
        psf_region = np.concatenate((np.arange(end_peak), tail))
        weights_psf = np.concatenate((np.full((end_peak,), 4.50), np.full_like(tail, 1)))
        markers_on_psf = [end_peak, *tail]

    psf_factor = curve_fit(scaling_gauss_func, psf_profile[psf_region], mixed_profile[psf_region],
                           sigma=weights_psf, bounds=bounds_psf)
    print("psf factor", psf_factor)

    star_profile = scaling_gauss_func(psf_profile, *psf_factor[0])
    star_profiles.append(star_profile)

    disk_profile = mixed_profile - star_profile

    fig_comp = plt.figure(figsize=(14, 7), num="Profiles " + profile[index])
    textax = plt.axes([0.5, 0.9, 0.3, 0.03], figure=fig_comp)
    textax.axis('off')
    textax.text(0, 0, profile[index], fontsize=18, ha='center')

    ax = fig_comp.add_subplot(1, 1, 1)
    ax.tick_params(labelsize=18)
    ax.plot(radi, cyc116_profile, '-', label="profile of cyc116", markevery=markers_on_nd4)
    ax.plot(radi, mixed_profile, '-', label="mixed profile", markevery=list(tail))
    nd4_equation = R"$({:.2})\cdot(ND4-({:.2}))$".format(*scaling_factor[0])
    ax.plot([], [], ' ', label=nd4_equation)
    ax.plot(radi, star_profile, '-C2', label="star profile", markevery=markers_on_psf)
    psf_equation = R"$({:.2})\cdot(gauss(PSF,{:.2})-({:.2}))$".format(*psf_factor[0])
    ax.plot([], [], ' ', label=psf_equation)
    ax.legend(fontsize='large', framealpha=1, loc=4)
    ax.set_yscale('log', nonposy='clip')
    ax.set_ylim(ymin=y_min)

    zoom_xax = (0, 60)

    axins = ax.inset_axes([0.35, 0.55, 0.5, 0.43])
    axins.semilogy(radi, cyc116_profile, '-', label="profile of cyc116", markevery=markers_on_nd4)
    axins.semilogy(radi, mixed_profile, '-', label="mixed profile", markevery=list(tail))
    axins.semilogy(radi, star_profile, '-', label="Star profile", markevery=markers_on_psf)
    axins.set_ylim(
        (0.9 * np.min(star_profile[zoom_xax[0]:zoom_xax[1]]), 1.5 * np.max(star_profile[zoom_xax[0]:zoom_xax[1]])))
    axins.set_xlim((-3, 60))
    ax.indicate_inset_zoom(axins)

    fig_sub = plt.figure(figsize=(16, 7), num="Disk " + profile[index])
    textax = plt.axes([0.5, 0.9, 0.3, 0.03], figure=fig_sub)
    textax.axis('off')
    textax.text(0, 0, "Subtraction in " + profile[index], fontsize=18, ha='center')

    ax = fig_sub.add_subplot(1, 1, 1)
    ax.tick_params(labelsize=18)
    line1, = ax.plot(radi, disk_profile, label="Reduced cyc116 profile")
    ax.set_xlim(xmin=-3.5, xmax=130)
    ax1 = ax.twinx()
    ax1.tick_params(labelsize=18)
    line2, = ax1.plot(x2[20:], qphi[20:], "C3", label="Qphi profile")
    ax.tick_params(axis='y', labelcolor="C0")
    ax1.tick_params(axis='y', labelcolor="C3")
    ax1.set_ylim(ymin=-40, ymax=1.1 * max(qphi[20:120]))
    ax.set_ylim(ymin=-500, ymax=1.1 * max(disk_profile[20:120]))
    line3 = ax.axhline(0, ls='--', c='k', alpha=0.5, label="zero")
    lines = [line1, line2, line3]
    align_yaxis(ax, 0, ax1, 0)
    ax.fill_between([32, 118], [-3000, -3000], [1000, 1000], alpha=0.2, color="gold")
    ax.legend(lines, [line.get_label() for line in lines], fontsize='x-large', framealpha=1, loc=1)

    if save:
        fig_comp.savefig(path + "/Profiles_" + profile[index] + ".png", dpi=150, bbox_inches='tight', pad_inches=0.1)
        fig_sub.savefig(path + "/Subtraction_" + profile[index] + ".png", dpi=150, bbox_inches='tight', pad_inches=0.1)

        param_file.write("\n" + profile[index] + "\n")
        param_file.write("Smart: {}\n".format(smart))
        param_file.write("ND4:\n")
        param_file.write("Region: {}\n".format(nd4_region))
        param_file.write("Weights: {}\n".format(weights_nd4))

        param_file.write("PSF:\n")
        param_file.write("Region: {}\n".format(psf_region))
        param_file.write("Weights: {}\n".format(weights_psf))
        param_file.write("\nScaling factor: {}\n".format(scaling_factor[0]))
        param_file.write(("PSF factor: {}\n".format(psf_factor[0])))

    disk_profile[disk_profile < 0] = 0
    count_disk = []
    count_qphi = []
    int_range = np.arange(-2, 3)
    for inner_wiggle in int_range:
        for outer_wiggle in int_range:
            count_disk.append(np.sum((disk_profile * circumference)[(32 + inner_wiggle):(118 + outer_wiggle)]))
            count_qphi.append(np.sum((qphi * circumference)[(32 + inner_wiggle):(118 + outer_wiggle)]))

    print("Counts fit: ", np.mean(count_disk), np.std(count_disk))
    print("Qphi counts: ", np.mean(count_qphi), np.std(count_qphi))
    print()

if save:
    print("File saved")
    print()
    param_file.close()

""" comparison """
print("--------- comparison ---------")
print()
cutoff = 80

_, cyc116_i = np.array(cyc116.azimuthal[0])
_, cyc116_r = np.array(cyc116.azimuthal[1])

comp_nd4_i = cyc116_i / scal_profiles[0]
comp_nd4_r = cyc116_r / scal_profiles[1]

comp_psf_i = scal_profiles[0] / star_profiles[0]
comp_psf_r = scal_profiles[1] / star_profiles[1]

fig = plt.figure(figsize=(14, 6))
textax = plt.axes([0.5, 0.9, 0.3, 0.03], figure=fig)
textax.axis('off')
textax.text(0, 0, "Comparison", fontsize=18, ha='center')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(labelsize=18)
ax.locator_params(axis='y', nbins=8)
ax.plot(np.arange(cutoff), comp_nd4_i[:cutoff], label="nd4_i")
ax.plot(np.arange(cutoff), comp_nd4_r[:cutoff], label="nd4_r")
ax.axhline(0.9, ls='--', c='k', alpha=0.125, zorder=-1)
ax.axhline(1.1, ls='--', c='k', alpha=0.125, zorder=-1)
ax.fill_between([start_int, end_int], [0.9, 0.9], [1.1, 1.1], alpha=0.2, color="gold")
ax.legend()
if save:
    fig.savefig(path + "/Comparison.png", dpi=150, bbox_inches='tight', pad_inches=0.1)

fig = plt.figure(figsize=(14, 6))
textax = plt.axes([0.5, 0.9, 0.3, 0.03], figure=fig)
textax.axis('off')
textax.text(0, 0, "Comparison", fontsize=18, ha='center')
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(labelsize=18)
ax.locator_params(axis='y', nbins=8)
ax.plot(np.arange(cutoff), comp_psf_i[:cutoff], label="nd4_i")
ax.plot(np.arange(cutoff), comp_psf_r[:cutoff], label="nd4_r")
ax.axhline(0.9, ls='--', c='k', alpha=0.125, zorder=-1)
ax.axhline(1.1, ls='--', c='k', alpha=0.125, zorder=-1)
ax.fill_between([start_peak, end_peak], [0.9, 0.9], [1.1, 1.1], alpha=0.2, color="gold")

if save:
    fig.savefig(path + "/Comparison2.png", dpi=150, bbox_inches='tight', pad_inches=0.1)

""" Aperture photometrie """

aperture_photometrie()

plt.show()
