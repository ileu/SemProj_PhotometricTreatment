from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.widgets import Slider, RadioButtons

# Set the grid
grid = gridspec.GridSpec(2, 1, height_ratios=[1, 2])

# Plot the sliders
axes_slider = plt.subplot(grid[0, 0])
slider = Slider(axes_slider, 'Channel', valmin=0, valmax=255, valinit=128)

# Plot the radio buttons
axes_button = plt.subplot(grid[1, 0])
button = RadioButtons(axes_button, ('Gradient', 'Channel'), active=1)

plt.tight_layout(h_pad=0)
plt.subplots_adjust(left=.2, right=.9)

def buttonupdate(val):
    if val == "Gradient":
        axes_slider.clear()
        slider.__init__(axes_slider, 'Gradient', valmin=-1, valmax=1, valinit=0)
        slider.on_changed(sliderupdate)
    else:
        axes_slider.clear()
        slider.__init__(axes_slider, 'Channel', valmin=0, valmax=255, valinit=128)
        slider.on_changed(sliderupdate)
    plt.gcf().canvas.draw_idle()

def sliderupdate(val):
    if slider.label.get_text() == 'Gradient':
        print("hulu")
        #do something depending on gradient value
        pass
    else:
        #do something depending on channel value
        print("HALLO")
        pass

# Register call-backs with widgets
slider.on_changed(sliderupdate)
button.on_clicked(buttonupdate)

plt.show()