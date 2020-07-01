import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons
from StarFunctions import StarImg

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
    axcolor = 'lavender'

    disk_map = star_data.radial[0][0]

    # GUI setup
    fig, ax = plt.subplots(figsize=(18, 11))

    ax.tick_params(labelsize=18)

    axinner = plt.axes([0.6, 0.85, 0.35, 0.03], facecolor=axcolor)
    axmiddle = plt.axes([0.6, 0.8, 0.35, 0.03], facecolor=axcolor)
    axouter = plt.axes([0.6, 0.75, 0.35, 0.03], facecolor=axcolor)

    sinner = Slider(axinner, 'Inner radius', 0, 50.0, valinit=ir0, valstep=pixel)
    smiddle = Slider(axmiddle, 'Annulus Size', 0, 75, valinit=mr0, valstep=pixel)
    souter = Slider(axouter, 'Background size', 0, 50.0, valinit=or0, valstep=pixel)

    rax = plt.axes([0.6, 0.60, 0.1, 0.1], facecolor=axcolor)
    rax.set_title("Select wave band:")
    radio = RadioButtons(rax, ('I\'-band', 'R\'-band'), active=0)
    for circle in radio.circles:
        circle.set_radius(0.07)

    resetax = plt.axes([0.85, 0.11, 0.08, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    textax = plt.axes([0.6, 0.5, 0.3, 0.03])
    textax.axis('off')

    textaxis = [textax.text(0, 0, "Disk", fontsize=14, fontweight='bold', color='blue'),
                textax.text(0, -1, "                     I'-band   R'-band"),
                textax.text(0, -2, "D"),
                textax.text(0, -3, "I"),
                textax.text(0, -4, "S"),
                textax.text(0, -5, "K")]

    # Plotting

    ax.set_ylim((362, 662))
    ax.set_xlim((362, 662))

    disk_plot = ax.imshow(disk_map, cmap='gray', vmin=-50, vmax=100)
    disk_mask_plot = ax.imshow(disk_map, cmap='gray', vmin=-50, vmax=100)

    plt.subplots_adjust(left=0.03, right=0.55, bottom=0.11)

    # interaction function

    def reset(event):
        sinner.reset()
        smiddle.reset()
        souter.reset()
        fig.canvas.draw_idle()

    def update(val=None):
        nonlocal rinner, rmiddle, router
        rinner = sinner.val
        rmiddle = smiddle.val
        router = souter.val

        disk_mask, total_counts, bg_counts, bg_avgs = star_data.mark_disk(rinner, rinner + rmiddle,
                                                                          rinner + rmiddle + router)

        ratio = bg_counts[0] / bg_counts[1]

        if ratio > 0:
            magnitude = 2.5 * np.log10(ratio)
        else:
            magnitude = np.nan

        textaxis[2].set_text("Total Count:  {:.0f}   {:.0f}".format(*total_counts))
        textaxis[3].set_text("Average BG:  {:.0f}   {:.0f}".format(*bg_avgs))
        textaxis[4].set_text("Counts wo BG:  {:.0f}   {:.0f}".format(*bg_counts))
        textaxis[5].set_text("Ratio I/R and magnitude:  {:.4f}   {:.2f}".format(ratio, 2.5 * np.log10(ratio)))

        disk_mask_plot.set_data(disk_mask)

        fig.canvas.draw()

    def change_band(label):
        nonlocal disk_map
        if label == 'I\'-band':
            disk_map = star_data.radial[0][0]
        elif label == 'R\'-band':
            disk_map = star_data.radial[1][0]
        else:
            raise ValueError("How is this even possible...")
        disk_plot.set_data(disk_map)
        fig.canvas.draw_idle()

    update()

    sinner.on_changed(update)
    smiddle.on_changed(update)
    souter.on_changed(update)

    button.on_clicked(reset)

    radio.on_clicked(change_band)

    # print("Ready")

    plt.show()
