# Thanks to me :P
# For More: +923440907874 (WhatsApp any kind of your questions)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import make_interp_spline
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'

# Load OpenSees result files
def load_file(fname):
    try:
        return np.loadtxt(fname)
    except Exception as e:
        print(f"Error loading {fname}: {e}")
        return np.zeros((1, 2))

base_disp = load_file("base_disp.out")
roof_disp = load_file("roof_disp.out")
base_force = load_file("base_force.out")
soil_disp = load_file("soil_disp.out")
soil_force = load_file("soil_force.out")

# Model parameters
n_stories = 22
h_story = 2.75
y_coords = np.linspace(0, n_stories * h_story, n_stories + 1)
time = roof_disp[:, 0]

# Generate deformed shape
def get_disp_profile(i):
    if i >= len(time): i = len(time) - 1
    u_base = base_disp[i, 1]
    u_roof = roof_disp[i, 1]
    drift = u_roof - u_base
    shape = np.linspace(-1, 1, n_stories + 1)
    curvature = 0.25 * drift * (1 - shape**2)
    return u_base + np.linspace(0, drift, n_stories + 1) + curvature

def smooth_line(x, y, resolution=200):
    spline = make_interp_spline(y, x, k=3)
    y_smooth = np.linspace(y.min(), y.max(), resolution)
    x_smooth = spline(y_smooth)
    return x_smooth, y_smooth

# Set up figure
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(3, 2)

# Main animation axis
ax_anim = fig.add_subplot(gs[:, 0])
line, = ax_anim.plot([], [], color='steelblue', lw=2, label="Structure")
roof_trace, = ax_anim.plot([], [], 'r--', alpha=0.3, lw=1, label="Roof Trace")
story_markers, = ax_anim.plot([], [], 'kD', markersize=4)
time_text = ax_anim.text(0.03, 0.97, '', transform=ax_anim.transAxes, fontsize=10, bbox=dict(facecolor='white', edgecolor='gray'))

# Roof displacement
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(time, roof_disp[:, 1], 'b', label='Roof Disp (m)')
roof_dot, = ax1.plot([], [], 'ro')
ax1.set_ylabel("Displacement (m)")
ax1.set_title("Roof Displacement", loc='left')
ax1.legend(loc='upper right')
max_roof = np.max(np.abs(roof_disp[:, 1]))
ax1.axhline(max_roof, ls='--', color='gray', lw=0.5)
ax1.annotate(f'Max: {max_roof:.2f} m',
             xy=(time[-1]*0.95, max_roof),
             xytext=(time[-1]*0.75, max_roof * 1.1),
             arrowprops=dict(arrowstyle='->', color='gray'),
             fontsize=8, color='gray')

# Base force
ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(time, base_force[:, 1], 'g', label='Base Shear (kN)')
base_dot, = ax2.plot([], [], 'ro')
ax2.set_ylabel("Force (kN)")
ax2.set_title("Base Shear Force", loc='left')
ax2.legend(loc='upper right')
max_force = np.max(np.abs(base_force[:, 1]))
ax2.axhline(max_force, ls='--', color='gray', lw=0.5)
ax2.annotate(f'Peak: {max_force:.1f} kN',
             xy=(time[-1]*0.95, max_force),
             xytext=(time[-1]*0.75, max_force * 1.1),
             arrowprops=dict(arrowstyle='->', color='gray'),
             fontsize=8, color='gray')

# Soil hysteresis
ax3 = fig.add_subplot(gs[2, 1])
ax3.plot(soil_disp[:, 1], soil_force[:, 1], 'k', label='Hysteresis')
soil_dot, = ax3.plot([], [], 'ro')
ax3.set_xlabel("Displacement (m)")
ax3.set_ylabel("Force (kN)")
ax3.set_title("Soil Spring Hysteresis", loc='left')
ax3.legend(loc='upper right')

# Add annotated arrows
idx_max = np.argmax(soil_force[:, 1])
idx_min = np.argmin(soil_force[:, 1])
ax3.annotate('Max force',
             xy=(soil_disp[idx_max, 1], soil_force[idx_max, 1]),
             xytext=(soil_disp[idx_max, 1] + 0.02, soil_force[idx_max, 1] * 1.05),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=8, color='red')

ax3.annotate('Min force',
             xy=(soil_disp[idx_min, 1], soil_force[idx_min, 1]),
             xytext=(soil_disp[idx_min, 1] - 0.05, soil_force[idx_min, 1] * 1.05),
             arrowprops=dict(arrowstyle='->', color='blue'),
             fontsize=8, color='blue')

# Limits
x_limit = max(abs(roof_disp[:, 1].max()), abs(base_disp[:, 1].min())) + 0.5
ax_anim.set_xlim(-x_limit, x_limit)
ax_anim.set_ylim(-2, h_story * (n_stories + 1))
ax_anim.set_xlabel("Displacement (m)")
ax_anim.set_ylabel("Height (m)")
ax_anim.set_title("Structural Animation with Floor Tracking", loc='left')
ax_anim.legend(loc='lower left')

# Floor number labels (optional)
add_floor_labels = True
if add_floor_labels:
    for y in y_coords:
        ax_anim.text(0.05, y, f"{int(y / h_story)}F", fontsize=7, va='center', color='gray')

x_trail, y_trail = [], []

# Animation functions
def init():
    line.set_data([], [])
    roof_trace.set_data([], [])
    story_markers.set_data([], [])
    roof_dot.set_data([], [])
    base_dot.set_data([], [])
    soil_dot.set_data([], [])
    time_text.set_text("")
    return line, roof_trace, roof_dot, base_dot, soil_dot, time_text, story_markers

def animate(i):
    disp_profile = get_disp_profile(i)
    x_smooth, y_smooth = smooth_line(disp_profile, y_coords)

    line.set_data(x_smooth, y_smooth)
    story_markers.set_data(disp_profile, y_coords)
    x_trail.append(disp_profile[-1])
    y_trail.append(y_coords[-1])
    roof_trace.set_data(x_trail, y_trail)

    roof_dot.set_data(time[i], roof_disp[i, 1])
    base_dot.set_data(time[i], base_force[i, 1])
    soil_dot.set_data(soil_disp[i, 1], soil_force[i, 1])
    time_text.set_text(f"Time = {time[i]:.2f} s")

    return line, roof_trace, roof_dot, base_dot, soil_dot, time_text, story_markers

# Run animation
ani = animation.FuncAnimation(
    fig, animate, frames=len(time),
    init_func=init, blit=True, interval=15, repeat=False
)

plt.tight_layout()
plt.show()
