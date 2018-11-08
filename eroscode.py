import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import initialisation


global C_D 
global C_H
global Q
global C_L
global alpha
global H
global rho_0
global R_E
global g_E
global rho_m
global Y
global KEs
global Heights
global ensemble
global initial_state
global analytical

KEs = []
Heights = []
rho_m = 3.3E3
Y = 2e6

def deg_to_rad(deg):
    """
    Returns an angle in radians
    for a given angle in degrees
    """
    return deg*np.pi / 180


def rad_to_degrees(rad):
    """
    Returns an angle in degrees
    for a given angle in radians
    """
    return rad*180 / np.pi


def dv(z, r, v, theta, m):
    """
    The ODE describing the rate of change in velocity
    """
    return (-C_D * rho_a(z) * area(r) * v ** 2) / (2 * m) + g_E * np.sin(theta)


def rho_a(z):
    """
    Returns the density for a given altitude
    """
    return rho_0 * np.exp(-z / H)


def dz(theta, v):
    """
    The ODE describing the rate of change in altitude
    """
    return -v * np.sin(theta)


def dtheta(theta, v, z, m, r):
    """
    The ODE describing the rate of change in angle
    of incidence relative to the horizon
    """

    term_1 = (g_E * np.cos(theta) / v)
    term_2 = (C_L * rho_a(z) * area(r) * v)/(2 * m)
    term_3 = (v * np.cos(theta)) / (R_E + z)
    
    return term_1 - term_2 - term_3


def dx(theta, v, z):
    """
    The ODE describing the rate of
    change in horizontal distance
    """
    return (v * np.cos(theta)) / (1 + z / R_E)


def dm(z, r, v):
    """
    The ODE describing the rate of change of mass
    """
    return (-C_H * rho_a(z) * area(r) * v ** 3) / (2 * Q)


def area(r):
    """
    Returns the cross sectional area of a
    circle for a given radius
    """
    return np.pi * r ** 2


def dr(v, z):
    """
    The ODE describing the rate of change of radius
    """
    return np.sqrt(7 / 2 * alpha * rho_a(z) / rho_m) * v


def efun(x, a, b, c):
    """
    Overlays an exponential function onto the graph
    of altitude versus time for visualisation purposes
    """
    return a*np.exp(-b*x)+c


def find_ke_max(data):
    t, v, m, theta, z, x, ke, r, burst_index, airburst_event = data
    ke_per_km = np.diff(ke) / np.diff(z/1000)/4.184e12
    ke_per_km = np.append(ke_per_km, ke_per_km[-1])
    ke_max_value = ke_per_km.max()
    ke_max_height = z[np.argmax(ke_per_km == ke_max_value)]
    plt.plot(ke_per_km, z)
    return ke_max_value, ke_max_height


def plot(t, v, m, z, KE, r, burst_index):
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
    ke_diff = np.diff(KE)
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


def ode_solver_pre_burst(t, state):
    """
    The set of ordinary differential equations
    describing the behavior of an asteroids
    reentry given that the stresses on it does not
    exceed the tensile strength of the asteroid
    """
    f = np.zeros_like(state)
    v, m, theta, z, x, r = state
    f[0] = dv(z, r, v, theta, m)
    f[1] = dm(z, r, v)
    if analytical is True:
        f[1] = 0

    f[2] = dtheta(theta, v, z, m, r)
    f[3] = dz(theta, v)
    f[4] = dx(theta, v, z)
    f[5] = 0
    return f

        
def ode_solver_post_burst(t, state):
    """
    The set of ordinary differential equations
    describing the behavior of an asteroid
    reentry given that the stresses on it does
    exceed the tensile strength of the asteroid
    """

    f = np.zeros_like(state)
    v, m, theta, z, x, r = state
    f[0] = dv(z, r, v, theta, m)
    f[1] = dm(z, r, v)

    f[2] = dtheta(theta, v, z, m, r)
    f[3] = dz(theta, v)
    f[4] = dx(theta, v, z)
    f[5] = dr(v, z)

    if analytical is True:
        f[1] = 0
        f[5] = 0

    return f


def main():
    state0 = initial_state

    """
    Place to execute the main code functions
    """
    # Initialising parameters and variables

    # Determining the time step and range that will be analysed
    t0 = 0
    tmax = 40.
    dt = 0.1
    t = np.arange(t0, tmax, dt)

    # solving the ODE for pre-burst conditions using Runga Kutta 45
    states = solve_ivp(ode_solver_pre_burst, (0, 1.1 * tmax), state0, t_eval=t, method='RK45')

    # Calculating the stresses felt by the asteroid
    v = np.array(states.y[0])
    z = np.array(states.y[3])
    tensile_stress = rho_a(z) * v ** 2

    # Calculating if the tensile stresses exceed the yield strength
    # And therefore if it is an airburst event
    burst_index = np.argmax(tensile_stress > Y)
    if burst_index == 0:
        print('Cratering Event')
        airburst_event = False
    else:
        print('Airburst Event')
        airburst_event = True

    # If the airburst occurs then rerun the ODEs from the
    # Point of burst and concatenate the two ODE solutions
    if airburst_event is True:
        t_new = t[burst_index]
        t2 = np.arange(t_new, tmax, dt)
        state0 = states.y[:, burst_index]

        states_2 = solve_ivp(ode_solver_post_burst,
                             (t_new, 1.1 * tmax), state0, t_eval=t2, method='RK45')

        solution = np.concatenate((states.y[:, 0:burst_index], states_2.y), axis=1)
    else:
        solution = states

    # plotting the quantities of interest
    v = solution[0]
    m = solution[1]
    theta = solution[2]
    z = solution[3]
    x = solution[4]
    ke = 0.5 * v ** 2 * m
    r = solution[5]

    final_state = t, v, m, theta, z, x, ke, r, burst_index, airburst_event

    return t, v, m, theta, z, x, ke, r, burst_index, airburst_event