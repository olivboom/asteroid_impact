import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# from scipy.integrate import odeint

# the issue is that the velocity is falling off too quickly
# should be decreasing inverse exponentially


def inital_parameter():
    """
    Creating the initial conditions for the quantities
    that will be solved in the ODE solver

    Returns
    -----------------
    An array containing the following quantities

    C_D  : float,  dimensionless
        Coefficient of drag.

    C_H : float, dimensionless
        The heat transfer coefficient

    Q : float, **********
        The heat of ablation constant

    C_L : float, dimensionless
        The coefficient of lift

    alpha : float, dimensionless
        Dispersion coefficient

    H : float, ************
        Atmospheric scale height

    rho_0 : float, kg / m ** 3
        Atmospheric density at sea level

    R_E : float, m
        The radius of the Earth

    r : float, m
        The radius of the asteroid

    g_E : float, m / 2 ** 2
        The acceleration due to gravity of the earth
    """

    global C_D
    global C_H
    global Q
    global C_L
    global alpha
    global H
    global rho_0
    global R_E
    global g_E

    C_D = 1.0
    C_H = 0.1
    Q = 1E7
    C_L = 1E-3
    alpha = 0.3
    H = 8e3
    rho_0 = 1.2
    R_E = 6.371e6
    g_E = 9.81


def initial_variables():
    """
    Creating the initial conditions for the quantities
    that will be solved in the ODE solver

    Returns
    -----------------
    An array containing the following quantities

    v_init : float
    The inital velocity

    m_init : float
        The intial mass

    theta_init : float
        The initial entry angle

    z_init : float
        The initial altitude that the asteroid is measured from

    x_init : float
        The initial condition zeroing the horizontal displacement ******** In which axes

    r_init : float
        The initial condition for the radius of the asteroid
    """

    v_init = 20e3
    m_init = 1000
    theta_init = 40 * np.pi / 180
    z_init = 100e3
    x_init = 0
    r_init = 3
    return np.array([v_init,
                     m_init,
                     theta_init,
                     z_init,
                     x_init,
                     r_init])


def ke(m, v):
    return 0.5 * m * v ** 2


def dv(z, r, v, theta, m):
    return (-C_D * rho_a(z) * area(r) * v ** 2) / (2 * m) + g_E * np.sin(theta)


def rho_a(z):
    return rho_0 * np.exp(-z / H)


def dz(theta, v):
    return -v * np.sin(theta)


def dtheta(theta, v, z, m, r):
    return (g_E * np.cos(theta) / v) - ((C_L * rho_a(z) * area(r) * v) /
                                        (2 * m)) - ((v * np.cos(theta)) / (R_E + z))


def dx(theta, v, z):
    return (v * np.cos(theta)) / (1 + z / R_E)


def dm(z, r, v):
    return (-C_H * rho_a(z) * area(r) * v ** 3) / (2 * Q)


def area(r):
    return np.pi * r ** 2


def volume(r):
    return 4 / 3 * np.pi * r ** 3


def dr(z, m, r):
    return 7 / 2 * alpha * rho_a(z) / (m / volume(r))


def graph_plot(t, states):
    plt.plot(t, states[:, 0])
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.show()

    plt.plot(t, states[:, 3])
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    plt.show()

    plt.plot(states[:, 3], states[:, 0])
    plt.xlabel('Altitude (m)')
    plt.ylabel('Velocity (m/s)')
    plt.show()

    plt.figure()
    plt.subplot(211)
    plt.plot(t, states[:, 3])
    plt.ylabel('Height')
    plt.subplot(212)
    plt.plot(t, states[:, 3])
    plt.ylabel('Speed')
    plt.xlabel('Time')
    plt.show()


def ode_solver(t, state):
    f = np.zeros_like(state)
    v, m, theta, z, x, r = state
    f[0] = dv(z, r, v, theta, m)
    f[1] = dm(z, r, v)
    f[2] = dtheta(theta, v, z, m, r)
    f[3] = dz(theta, v)
    f[4] = dx(theta, v, z)
    f[5] = dr(z, m, r)
    return f


def main():
    inital_parameter()
    state0 = initial_variables()
    dt = 0.1
    t = np.arange(0.0, 40.0, dt)

    # =============================================================================
    #     SOLVE_IVP
    # =============================================================================

    states = solve_ivp(ode_solver, (0, 50), state0, t_eval=t, method='RK45')  # need to fix the f tomorrow
    v = states.y[0]
    m = states.y[1]
    theta = states.y[2]
    z = states.y[3]
    x = states.y[4]
    r = states.y[5]
    KE = 0.5 * m * v ** 2

    plt.figure()
    plt.subplot(311)
    plt.plot(t, z)
    plt.ylabel("Height")

    plt.subplot(312)
    plt.ylabel("Speed")
    plt.plot(t, v)
    plt.plot

    plt.subplot(313)
    plt.ylabel("Energy")
    plt.xlabel("Time")

    plt.plot(t, KE)
    plt.show()

    # states = odeint(ode_solver, state0, t, tfirst=True)
    # graph_plot(t, states)


main()
# if __name__ == "main":
#     main()


'''
=============================================================================
ODEINT
=============================================================================

states = odeint(f, state0, t, tfirst = True)

v = states[:, 0]
m = states[:, 1]
theta = states[:, 2]
z = states[:, 3]
x = states[:,4]
r = states[:,5]


KE = 0.5 * m * v**2


plt.figure()
plt.subplot(211)
plt.plot(t,z)
plt.ylabel("Height")

plt.subplot(212)
plt.ylabel("Speed")
plt.xlabel("Time")
plt.plot(t,v)

'''
