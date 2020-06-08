import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.widgets import Slider, Button, RadioButtons
from StarFunctions import StarImg, OOI, ballestero

plt.rcParams["image.origin"] = 'lower'


# functions

def start(star_data: StarImg):
    # Start parameters

    ir0 = 16  # maybe 21
    or0 = 12
    rinner = ir0
    router = or0
    pixel = 1.0
    waveband = 'I'
    axcolor = 'lavender'

    textaxes = []

    fig, ax = plt.subplots(figsize=(18, 11))

    Star_Data: StarImg = None
    StarPlot = None

    axinner = plt.axes([0.58, 0.85, 0.35, 0.03], facecolor=axcolor)
    axouter = plt.axes([0.58, 0.8, 0.35, 0.03], facecolor=axcolor)

    sinner = Slider(axinner, 'Aperture Size', 0, 35.0, valinit=ir0, valstep=pixel)
    souter = Slider(axouter, 'Annulus SIze', 0, 20.0, valinit=or0, valstep=pixel)

    rax = plt.axes([0.58, 0.65, 0.1, 0.1], facecolor=axcolor)
    rax.set_title("Select wave band:")
    radio = RadioButtons(rax, ('I\'-band', 'R\'-band'), active=0)
    # GUI setup

    Star_Data = star_data

    starmap = Star_Data.show_objects(ir0, or0)

    StarPlot = ax.imshow(starmap)

    plt.subplots_adjust(left=0.01, right=0.54, bottom=0.11)

    for circle in radio.circles:
        circle.set_radius(0.07)

    resetax = plt.axes([0.85, 0.11, 0.08, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    for index, obj in enumerate(Star_Data.objects):
        textax = plt.axes([0.65 - 0.1 * (-1) ** index, 0.6 - 0.2 * np.floor(index / 2), 0.3, 0.03])
        textax.axis('off')
        obj.count = [ir0, or0]
        total_counts, bg_counts, bg_avgs = obj.count_pixel(Star_Data, filter_val=Star_Data.filter_reduction)
        ratio = bg_counts[0] / bg_counts[1]
        magnitude = 2.5 * np.log10(ratio)
        textaxis = [textax.text(0, 0, obj.name, fontsize=14, fontweight='bold', color='blue'),
                    textax.text(0, -1, "                     I'-band   R'-band"),
                    textax.text(0, -2, "Total Count:  {:.0f}   {:.0f}".format(*total_counts)),
                    textax.text(0, -3, "Average BG:  {:.0f}   {:.0f}".format(*bg_avgs)),
                    textax.text(0, -4, "Counts wo BG:  {:.0f}   {:.0f}".format(*bg_counts)),
                    textax.text(0, -5, "Ratio I/R and magnitude:  {:.4f}   {:.2f}".format(ratio, magnitude))]
        textaxes.append(textaxis)

    # interaction function

    def update_count():
        for text, obj in zip(textaxes, Star_Data.objects):
            total_counts, bg_counts, bg_avgs = obj.count_pixel(Star_Data, filter_val=Star_Data.filter_reduction)
            ratio = bg_counts[0] / bg_counts[1]
            text[2].set_text("Total Count:  {:.0f}   {:.0f}".format(*total_counts))
            text[3].set_text("Average BG:  {:.0f}   {:.0f}".format(*bg_avgs))
            text[4].set_text("Counts wo BG:  {:.0f}   {:.0f}".format(*bg_counts))
            text[5].set_text("Ratio I/R and magnitude:  {:.4f}   {:.2f}".format(ratio, 2.5 * np.log10(ratio)))

    def reset(event):
        sinner.reset()
        souter.reset()
        fig.canvas.draw_idle()

    def update(val):
        old_rin = sinner.val
        old_rout = souter.val

        StarPlot.set_data(Star_Data.show_objects(old_rin, old_rout, band=waveband))
        update_count()

        fig.canvas.draw()

    def change_band(band, label):
        if label == 'I\'-band':
            band = 'I'
            StarPlot.set_data(Star_Data.show_objects(rinner, router, band=band))
        elif label == 'R\'-band':
            band = 'R'
            StarPlot.set_data(Star_Data.show_objects(rinner, router, band=band))
        else:
            raise ValueError("How is this even possible...")

        fig.canvas.draw_idle()

    sinner.on_changed(update)
    souter.on_changed(update)

    button.on_clicked(reset)

    radio.on_clicked(change_band)

    print("Ready")


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
