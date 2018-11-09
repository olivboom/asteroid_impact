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
global KEs
global Heights
global ensemble
global initial_state
global analytical
global tol
global final_state

tol = None

KEs = []
Heights = []
rho_m = 3.3E3

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



def find_ke_max(data):
    t, v, m, theta, z, x, ke, r, burst_index, airburst_event = data
    
    z_diff = np.diff(z)
    z_diff = np.append(z_diff, z_diff[-1])
    ke_per_km = np.diff(ke) / np.diff(z/1000)/4.184e12
    ke_per_km = np.append(ke_per_km, ke_per_km[-1])
    ke_max_value = ke_per_km.max()
    ke_max_height = z[np.argmax(ke_per_km == ke_max_value)]
#    plt.plot(ke_per_km, z)
    return ke_max_value, ke_max_height




def ode_solver_pre_burst(t, state):
    """
    The set of ordinary differential equations
    describing the behavior of an asteroids
    reentry given that the stresses on it does not
    exceed the tensile strength of the asteroid
    """
    f = np.zeros_like(state)
#    print(state)
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


def main(v,m,theta,z,x,r,Y):
    #state0 = initial_state
    state0=v,m,theta,z,x,r
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
    states = solve_ivp(ode_solver_pre_burst, (0, 1.1 * tmax), state0, t_eval=t, method='RK45', atol=tol, rtol=tol)

    # Calculating the stresses felt by the asteroid
    v = np.array(states.y[0])
    z = np.array(states.y[3])
    tensile_stress = rho_a(z) * v ** 2

    # Calculating if the tensile stresses exceed the yield strength
    # And therefore if it is an airburst event
    burst_index = np.argmax(tensile_stress > Y)
    if burst_index == 0:
#        #print('Cratering Event')
        airburst_event = False
    else:
#        #print('Airburst Event')
        airburst_event = True

    # If the airburst occurs then rerun the ODEs from the
    # Point of burst and concatenate the two ODE solutions
    if airburst_event is True:
        t_new = t[burst_index]
        t2 = np.arange(t_new, tmax, dt)
        state0 = states.y[:, burst_index]

        states_2 = solve_ivp(ode_solver_post_burst,
                             (t_new, 1.1 * tmax), state0, t_eval=t2, method='RK45', atol=tol, rtol=tol)

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

    final_state = t, v, m, theta, z, x, ke, r, burst_index, airburst_event, Y

    return final_state