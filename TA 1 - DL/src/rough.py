import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# create figure and axis objects
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# set up data
t = np.linspace(0, 2 * np.pi, 100)
x = np.sin(t)
y = np.cos(t)
z = np.zeros_like(t)

# create the initial plot
line, = ax.plot(x, y, z, marker='o', markersize=1, color='red')

# add text annotation
text = ax.text(x[-1], y[-1], z[-1], '', color='black')


# define the update function
def update(num):
    line.set_data(x[:num], y[:num])
    line.set_3d_properties(z[:num])
    text.set_text(f'Frame {num}')
    text.set_position((x[num], y[num], z[num]))
    return line, text


# create animation
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)

# show the plot
plt.show()
