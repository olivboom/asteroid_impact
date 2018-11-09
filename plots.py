# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 19:03:03 2018

@author: Hamed
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def efun(x, a, b, c):
    """
    Overlays an exponential function onto the graph
    of altitude versus time for visualisation purposes
    """
    return a*np.exp(-b*x)+c


def plot_5_graphs(data):
    """
    Plots a series of graphs that are solutions to the
    set of ODEs for given initial conditions.

    Returns
    -------------
    Plot with subplots for
    Time vs Altitude
    Speed vs Altitude
    Altiude vs Energy lost per unit height
    Mass vs Altitude
    Diameter vs Altitude
    """
    t, v, m, theta, z, x, ke, r, burst_index, airburst_event = data

    fig = plt.figure(figsize=(8, 15))
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.7)

    popt, pcov = curve_fit(efun, t, z / 1e3)
    yy = efun(t, *popt)

    ax1 = fig.add_subplot(321)
    ax1.plot(t, z / 1e3)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Altitude [km]")
    ax1.plot(t, yy, '--r', label='{:.2f}*exp(-{:.2f}x)+{:.2f}'.format(popt[0], popt[1], popt[2]))
    ax1.legend()
    ax1.grid()

    ax2 = fig.add_subplot(322)
    ax2.set_xlabel("Speed [km/s]")
    ax2.set_ylabel("Altitude [km]")
    ax2.plot(v / 1e3, z / 1e3)
    ax2.set_xlim(ax2.get_xlim()[::-1])
    ax2.grid()

    ax3 = fig.add_subplot(323)
    ax3.set_ylabel("Altitude [km]")
    ax3.set_xlabel("Energy_Loss [kt/km]")
    ke_diff = np.diff(ke)
    z_diff = np.diff(z)
    ke_unit = abs(ke_diff / z_diff) * 1e3 / 4.18E12
    ax3.plot(ke_unit, z[:-1] / 1e3)
    ax3.grid()

    ax4 = fig.add_subplot(324)
    ax4.set_xlabel("Mass [kg]")
    ax4.set_ylabel("Altitude [km]")
    ax4.plot(m / 1e3, z / 1e3)
    ax4.set_xlim(ax4.get_xlim()[::-1])
    ax4.grid()

    ax5 = fig.add_subplot(325)
    ax5.set_xlabel("Diameter [m]")
    ax5.set_ylabel("Altitude [km]")
    ax5.plot((r * 2), z / 1e3)
    ax5.set_xlim(ax5.get_xlim()[::-1])
    ax5.grid()

    z_burst = z[ke_unit.argmax()] / 1e3
    ax3.axhline(y=z_burst, color='r', linestyle='--', linewidth=1)
    ax3.axvline(x=ke_unit.max(), color='r', linestyle='--', linewidth=1)
    ax3.annotate('{:.2E}'.format(z_burst), xy=(0, 2 * z_burst), color='r')
    ax3.annotate('{:.2E}'.format(np.max(ke_unit)), xy=(ke_unit.max() + ke_unit.max() / 100, 8 * z_burst),
                 color='r', rotation=90)
    ax3.annotate('*', xy=(ke_unit[burst_index], z[burst_index] / 1e3))
    plt.show()

def ensemble_scatter(KE,height):
    
    tuples = zip(KE,height)
    a = list(tuples)
    x,y = np.array(a)[:,0], np.array(a)[:,1]

    plt.plot(x,y,'.')
    plt.xlabel('KE')
    plt.ylabel('Height')
    plt.show()

def analytical_comparison(analytical, numerical):
    x1,y1 = analytical
    t, v, m, theta, z, x, ke, r, burst_index, airburst_event = numerical

    
    ke_diff = np.diff(ke)
    z_diff = np.diff(z)
    ke_unit = abs(ke_diff / z_diff) * 1e3 / 4.18E12
    plt.plot(x1,y1, '.')
    plt.grid()
    plt.show()

    
    