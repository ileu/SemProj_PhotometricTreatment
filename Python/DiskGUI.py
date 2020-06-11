import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.widgets import Slider, Button, RadioButtons
from StarFunctions import StarImg, OOI, ballestero

plt.rcParams["image.origin"] = 'lower'


# functions

def start(star_data: StarImg):
    # Start parameters
    ir0 = 28
    mr0 = 65
    or0 = 31
    rinner = ir0
    rmiddle = mr0
    router = or0
    pixel = 1.0
    waveband = 'I'
    axcolor = 'lavender'

    fig, ax = plt.subplots(figsize=(18, 11))

    Star_Data: StarImg = None
    StarPlot = None

    axinner = plt.axes([0.58, 0.85, 0.35, 0.03], facecolor=axcolor)
    axmiddle = plt.axes([0.58, 0.8, 0.35, 0.03], facecolor=axcolor)
    axouter = plt.axes([0.58, 0.75, 0.35, 0.03], facecolor=axcolor)

    sinner = Slider(axinner, 'Inner radius', 0, 50.0, valinit=ir0, valstep=pixel)
    smiddle = Slider(axmiddle, 'Annulus Size', 0, 75, valinit=mr0, valstep=pixel)
    souter = Slider(axouter, 'Background size', 0, 50.0, valinit=or0, valstep=pixel)

    rax = plt.axes([0.58, 0.60, 0.1, 0.1], facecolor=axcolor)
    rax.set_title("Select wave band:")
    radio = RadioButtons(rax, ('I\'-band', 'R\'-band'), active=0)
    # GUI setup

    Star_Data = star_data
    Star_Data.calc_radial_polarization()
    diskmap, total_counts, bg_counts, bg_avgs = Star_Data.mark_disk(ir0, ir0 + mr0, ir0 + mr0 + or0)
    print(diskmap, total_counts, bg_counts, bg_avgs)
    ax.margins(x=350, y=350)
    StarPlot = ax.imshow(diskmap, cmap='viridis', vmin=-50, vmax=100)

    plt.subplots_adjust(left=0.01, right=0.54, bottom=0.11)

    for circle in radio.circles:
        circle.set_radius(0.07)

    resetax = plt.axes([0.85, 0.11, 0.08, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    textax = plt.axes([0.58, 0.5, 0.3, 0.03])
    textax.axis('off')
    ratio = bg_counts[0] / bg_counts[1]
    textaxis = [textax.text(0, 0, "Disk", fontsize=14, fontweight='bold', color='blue'),
                textax.text(0, -1, "                     I'-band   R'-band"),
                textax.text(0, -2, "Total Count:  {:.0f}   {:.0f}".format(*total_counts)),
                textax.text(0, -3, "Average BG:  {:.0f}   {:.0f}".format(*bg_avgs)),
                textax.text(0, -4, "Counts wo BG:  {:.0f}   {:.0f}".format(*bg_counts)),
                textax.text(0, -5, "Relativ reduction:  {:.4f}   {:.4f}".format(*(bg_counts / total_counts))),
                textax.text(0, -6, "Ratio I/R:  {:.4f}  ".format(ratio))]

    # interaction function

    def reset(event):
        sinner.reset()
        smiddle.reset()
        souter.reset()
        fig.canvas.draw_idle()

    def update(val):
        nonlocal rinner, rmiddle, router, waveband
        rinner = sinner.val
        rmiddle = smiddle.val
        router = souter.val

        new_plot, total_counts, bg_counts, bg_avgs = Star_Data.mark_disk(rinner, rinner + rmiddle,
                                                                         rinner + rmiddle + router, band=waveband)
        ratio = bg_counts[0] / bg_counts[1]
        textaxis[2].set_text("Total Count:  {:.0f}   {:.0f}".format(*total_counts))
        textaxis[3].set_text("Average BG:  {:.0f}   {:.0f}".format(*bg_avgs))
        textaxis[4].set_text("Counts wo BG:  {:.0f}   {:.0f}".format(*bg_counts))
        textaxis[5].set_text("Relativ reduction:  {:.4f}   {:.4f}".format(*(bg_counts / total_counts))),
        textaxis[6].set_text("Ratio I/R and magnitude:  {:.4f}   {:.2f}".format(ratio, 2.5 * np.log10(ratio)))

        StarPlot.set_data(new_plot)
        print("draw")
        fig.canvas.draw()

    def change_band(label):
        print("Click")
        nonlocal waveband
        if label == 'I\'-band':
            waveband = 'I'
            update(None)
        elif label == 'R\'-band':
            waveband = 'R'
            update(None)
        else:
            raise ValueError("How is this even possible...")

        fig.canvas.draw_idle()

    sinner.on_changed(update)
    smiddle.on_changed(update)
    souter.on_changed(update)

    button.on_clicked(reset)

    radio.on_clicked(change_band)

    # print("Ready")

    plt.show()
