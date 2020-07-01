import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons
from StarFunctions import StarImg

plt.rcParams["image.origin"] = 'lower'


def start(star_data: StarImg):
    # Start parameters
    ir0 = 16
    or0 = 12
    rinner = ir0
    router = or0
    a = 1e-2
    pixel = 1.0
    axcolor = 'lavender'

    textaxes = []

    star_map = star_data.get_i_img()[0]
    star_map = np.log10(a * star_map + 1)
    obj_names = [obj.name for obj in star_data.objects]

    # GUI setup

    fig, ax = plt.subplots(figsize=(18, 11))

    ax.tick_params(labelsize=18)

    axinner = plt.axes([0.6, 0.85, 0.35, 0.03], facecolor=axcolor)
    axouter = plt.axes([0.6, 0.8, 0.35, 0.03], facecolor=axcolor)

    sinner = Slider(axinner, 'Aperture Size', 0, 35.0, valinit=ir0, valstep=pixel)
    souter = Slider(axouter, 'Annulus SIze', 0, 20.0, valinit=or0, valstep=pixel)

    rax = plt.axes([0.6, 0.65, 0.1, 0.1], facecolor=axcolor)
    rax.set_title("Select wave band:")
    radio = RadioButtons(rax, ('I\'-band', 'R\'-band'), active=0)
    for circle in radio.circles:
        circle.set_radius(0.07)

    resetax = plt.axes([0.87, 0.05, 0.08, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    for index, name in enumerate(obj_names):
        textax = plt.axes([0.7 - 0.1 * (-1) ** index, 0.62 - 0.21 * np.floor(index / 2), 0.3, 0.03])
        textax.axis('off')
        textaxis = [textax.text(0, 0, name, fontsize=14, fontweight='bold', color='blue'),
                    textax.text(0, -1, "                     I'-band   R'-band"),
                    textax.text(0, -2, "S"),
                    textax.text(0, -3, "T"),
                    textax.text(0, -4, "A"),
                    textax.text(0, -5, "R")]
        textaxes.append(textaxis)

    # Plotting
    star_plot = ax.imshow(star_map, cmap='gray')
    star_mask_plot = ax.imshow(star_map, cmap='gray')

    plt.subplots_adjust(left=0.03, right=0.55, bottom=0.11)

    # interaction function

    def reset(event):
        sinner.reset()
        souter.reset()
        fig.canvas.draw_idle()

    def update(val=None):
        nonlocal rinner, router
        rinner = sinner.val
        router = souter.val

        star_mask, total_counts, bg_counts, bg_avgs = star_data.mark_objects(rinner, rinner + router)

        for index, text in enumerate(textaxes):
            # print(obj.name)
            # print(err)
            ratio = bg_counts[index][0] / bg_counts[index][1]
            # ratio_err = (err[index][0] / bg_counts[index][1]) ** 2
            # ratio_err += (err[index][1] * bg_counts[index][0] / bg_counts[index][1] ** 2) ** 2
            # ratio_err = np.sqrt(ratio_err)

            if ratio > 0:
                magnitude = 2.5 * np.log10(ratio)
            else:
                magnitude = np.nan

            # print(ratio_err)
            text[2].set_text("Total Count:  {:.4}   {:.4}".format(*total_counts[index]))
            text[3].set_text("Average BG:  {:.4}   {:.4}".format(*bg_avgs[index]))
            text[4].set_text("Counts wo BG:  {:.4}   {:.4}".format(*bg_counts[index]))
            text[5].set_text("Ratio I/R and magnitude:  {:.4}   {:.2}".format(ratio, magnitude))

        star_mask_plot.set_data(star_mask)

        fig.canvas.draw()

    def change_band(label):
        nonlocal star_map
        # print("Click")
        if label == 'I\'-band':
            star_map = star_data.get_i_img()[0]
        elif label == 'R\'-band':
            star_map = star_data.get_r_img()[0]
        else:
            raise ValueError("How is this even possible...")

        star_map = np.log10(a * star_map + 1)
        star_plot.set_data(star_map)
        fig.canvas.draw_idle()

    update()

    sinner.on_changed(update)
    souter.on_changed(update)

    button.on_clicked(reset)

    radio.on_clicked(change_band)

    # print("Ready")

    plt.show()
