import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def create_r_mesh(ri,rf,total_points):
    """"
    This function dynamically adjusts the mesh to ensure smooth coverage over both
    short and wide ranges. For ranges where the order of magnitude difference between
    `ri` and `rf` is 2 or less, it uses linear spacing. For wider ranges, it combines
    linear spacing (from `ri` to a transitional radius `r_trans`) and logarithmic
    spacing (from `r_trans` to `rf`).

    Parameters:
    - ri (float): The starting radius value.
    - rf (float): The ending radius value.
    - total_points (int): The total number of points desired in the mesh.
    """

    # Dynamically determine transition point based on the magnitude difference
    magnitude_difference = np.log10(rf) - np.log10(ri)
    if magnitude_difference <= 2:
        # If the range is not wide, use linear spacing
        r = np.linspace(ri, rf, total_points)
    else:
        # For wider ranges, use a combination of linear and logarithmic
        # Determine transition point as a power of 10 just above ri
        r_trans = 10**np.ceil(np.log10(ri))
        
        # Adjust the number of points for linear and logarithmic portions based on their range
        linear_fraction = (np.log10(r_trans) - np.log10(ri)) / (np.log10(rf) - np.log10(ri))
        num_linear_points = int(linear_fraction * total_points)
        num_log_points = total_points - num_linear_points
        
        # Create linear spacing from ri to r_trans
        r_linear = np.linspace(ri, r_trans, num_linear_points, endpoint=False)
        
        # Create logarithmic spacing from r_trans to rf
        r_log = np.logspace(np.log10(r_trans), np.log10(rf), num_log_points, endpoint=True)
        
        r = np.concatenate((r_linear, r_log))
    return r

def plot_something(x, y, y_label, radius=None, x_label=r'$r(km)$', y_scale=None, debug=False):
    plt.figure(figsize=(8, 4))
    plt.plot(x/1e5, y, color='blue')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if debug:
       plt.xlim(1e-1,2e2) 
    else:
        plt.xscale('log')
    if y_scale=='log':
        plt.yscale('log')
    plt.axvline(x=radius/1e5, linestyle='--', color='grey', linewidth=0.4,zorder=-1)
    plt.axhline(y=0, linestyle='--', color='grey', linewidth=0.4,zorder=-1)
    plt.xlim(1e-4,x[-1]/1e5)
    return plt.show()

################# to plot one variale on the full integration length #################
def plot_full(x_in, x_out, radius, y_in, y_out, y_label, y_scale=None, debug=False):
    plt.figure(figsize=(8, 4))
    plt.plot(x_in/1e5, y_in, color='red')
    plt.plot(x_out/1e5, y_out, color='blue')
    plt.xlabel(r'$r(km)$')
    plt.ylabel(f"${y_label}$")
    if debug:
        plt.xlim(8,20) 
    else:
        plt.xlim(1e-3,x_out[-1]/1e5)
        plt.xscale('log')
    if y_scale=='log':
        plt.yscale('log')
    plt.axvline(x=radius/1e5, linestyle='--', color='grey', linewidth=0.4,zorder=-1)
    plt.axhline(y=0, linestyle='--', color='grey', linewidth=0.4,zorder=-1)
    return plt.show()

################# to plot two variables side by side #################
def plot_full_side_by_side(x_in, x_out, radius, y_in, y_out, y_label, subplot_index, y_scale=None, debug=False):
    # Assuming a 1x2 subplot layout for side-by-side plots
    ax = plt.subplot(1, 2, subplot_index)
    ax.plot(x_in/1e5, y_in, color='red')
    ax.plot(x_out/1e5, y_out, color='blue')
    ax.set_xlabel(r'$r(km)$')
    ax.set_ylabel(f"${y_label}$")
    if debug:
        ax.set_xlim(8,20)
    else:
        ax.set_xlim(1e-3,x_out[-1]/1e5)
        ax.set_xscale('log')
    if y_scale == 'log':
        ax.set_yscale('log')
    ax.axvline(x=radius/1e5, linestyle='--', color='grey', linewidth=0.4, zorder=-1)
    if y_in[0]!=1:
        ax.axhline(y=0, linestyle='--', color='grey', linewidth=0.4, zorder=-1)