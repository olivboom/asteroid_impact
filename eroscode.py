import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

def inital_parameter():
    global C_D
    global C_H
    global Q
    global C_L
    global alpha
    global H
    global rho_0
    global R_E
    global r
    global g_E
    C_D = 1  # Drag coefficient
    C_H = 0.1  # Heat transfer coefficient
    Q = 1E7  # Heat of ablation
    C_L = 1E-3  # Lift coefficient
    alpha = 0.3  # Dispersion coefficient
    H = 8000  # Atmospheric scale height
    rho_0 = 1.2  # Atmospheric density at sea level
    R_E = 6.371e6  # Radius of Earth
    r = 3  # Radius of asteroid
    g_E = 9.81  # gravity constant on Earth


def initial_variables():
    v_init = 20e3  # initial body axes velocity of asteroid
    m_init = 1000 # initial mass in kg
    theta_init = 90 * np.pi / 180
    z_init = 100e3 # initial altitude in metres
    x_init = 0  # initial horizontal displacement
    r_init = 3  # initial radius of the asteroid
#    m_init = 4/3 * np.pi * r_init**3 * 3300

    '''
    Where to include data around initial volume and mass
    V_init = Volume(r)  # initial volume
    m_init = V_init * rho_m_init
    rho_m_init = 3300  # initial asteroid density
    '''

    return np.array([v_init, m_init,
                     theta_init, z_init, x_init, r_init])


def KE(m, v):
    return 0.5 * m * v ** 2


def dv(z, r, v, theta, m):
    return ((-C_D * rho_a(z) * A(r) * v ** 2) / (2 * m) + g_E * np.sin(theta)) 


def rho_a(z):
    return rho_0 * np.exp(-z / H)


def dz(theta, v):
    return -v * np.sin(theta)


def dtheta(theta, v, z, m, r):
    return ((g_E * np.cos(theta) / v) - ((C_L * rho_a(z) * A(r) * v) / (2 * m)) - ((v * np.cos(theta)) / (R_E + z)))


def dx(theta, v, z):
    return (v * np.cos(theta)) / (1 + (z / R_E))


def dm(z, v):
    return (-C_H * rho_a(z) * A(r) * v ** 3) / (2 * Q)


def A(r):
    return np.pi * r ** 2


def Volume(r):
    return 4 / 3 * np.pi * r ** 3


def dr(z, m, r):
    return (7 / 2 * alpha * rho_a(z) / (m/Volume(r))) # need to decide whether to store volume



def f(t, state):
    f = np.zeros_like(state)
    v, m, theta, z, x, r = state
    f[0] = dv(z, r, v, theta, m)
    f[1] = dm(z, v)
    f[2] = dtheta(theta, v, z, m, r)
    f[3] = dz(theta, v)
    f[4] = dx(theta, v, z)
    f[5] = dr(z, m, r)
    return f


def main():
    
# =============================================================================
#     SOLVE_IVP
# =============================================================================
    inital_parameter()
    state0 = initial_variables()
    dt = 0.1
    t = np.arange(0.0, 40.0, dt)
    states = solve_ivp(f, (0,50), state0, t_eval = t, method = 'RK45')
    v = states.y[0]
    m = states.y[1]
    theta = states.y[2]
    z = states.y[3]
    x = states.y[4]
    r = states.y[5]
    KE = 0.5 * m * v**2

    plt.figure()
    plt.subplot(311)
    plt.plot(t,z)
    plt.ylabel("Height")
    
    plt.subplot(312)
    plt.ylabel("Speed")
    plt.plot(t,v)
    plt.plot
    
    plt.subplot(313)
    plt.ylabel("Energy")
    plt.xlabel("Time")

    plt.plot(t,KE)

# =============================================================================
# ODEINT
# =============================================================================
    
#    states = odeint(f, state0, t, tfirst = True)
#    
#    v = states[:, 0]
#    m = states[:, 1]
#    theta = states[:, 2]
#    z = states[:, 3]
#    x = states[:,4]
#    r = states[:,5]
#    
#    
#    KE = 0.5 * m * v**2
#    
#    
#    plt.figure()
#    plt.subplot(211)
#    plt.plot(t,z)
#    plt.ylabel("Height")
#    
#    plt.subplot(212)
#    plt.ylabel("Speed")
#    plt.xlabel("Time")
#    plt.plot(t,v)


main()
if __name__ == "main":
    main()
