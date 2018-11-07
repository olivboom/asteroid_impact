import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

def inital_parameter():
    """
    Creating the initial conditions for the quantities
    that will be solved in the ODE solver

    Returns
    -----------------
    An array containing the following quantities

    C_D  : float,  dimensionless
        Coefficient of drag.

    C_H : float, W/(m**2*K)
        The heat transfer coefficient

    Q : float, J/kg
        The heat of ablation constant

    C_L : float, dimensionless
        The coefficient of lift

    alpha : float, dimensionless
        Dispersion coefficient

    H : float, m
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
    global rho_m
    global Y

    C_D = 1.0
    C_H = 0.1
    Q = 1E7
    C_L = 1E-3
    alpha = 0.3
    H = 8e3
    rho_0 = 1.2
    R_E = 6.371e6
    g_E = 9.81
    rho_m=3.3E3
    Y = 2E6
    
def deg_to_rad(deg):
    return deg*np.pi/ 180


def rad_to_degrees(rad):
    return rad*180/np.pi

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

    v_init = 19e3
    m_init = 12e6
    theta_init = deg_to_rad(20)
    z_init = 100e3
    x_init = 0
    r_init = 19.5/2
    return np.array([v_init,
                     m_init,
                     theta_init,
                     z_init,
                     x_init,
                     r_init])

def geth(r,m):
    return m/(np.pi*r**2*rho_m)

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


def volume(r, h):
    return np.pi*r**2*h


def dr(z, m, r):
    h=geth(r,m)
    return 7 / 2 * alpha * rho_a(z) / (m / volume(r, h))


def density(m,r):
    return m/volume(r)	


def mass(rho,r):
    return rho*volume(r)


def efun(x, a, b, c):
    return a*np.exp(-b*x)+c


def plot(t, v, z, KE, burst_index):
    fig = plt.figure(figsize=(8, 15))
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.7)

    # fig1=plt.subplot(311)
    ax1 = fig.add_subplot(311)

    ax1.plot(t, z / 1e3)
    ax1.set_xlabel("Time[s]")
    ax1.set_ylabel("Altitude[km]")

    popt, pcov = curve_fit(efun, t, z / 1e3, p0=(1, 1e-6, 1))
    yy = efun(t, *popt)
    ax1.plot(t, yy, '--r', label='{:.2f}*exp(-{:.2f}x)+{:.2f}'.format(popt[0], popt[1], popt[2]))
    ax1.legend()

    ax2 = fig.add_subplot(312)
    ax2.set_xlabel("Speed[km/s]")
    ax2.set_ylabel("Altitude[km]")
    ax2.plot(v / 1e3, z / 1e3)
    ax2.set_xlim(ax2.get_xlim()[::-1])

    ax3 = fig.add_subplot(313)
    ax3.set_ylabel("Altitude[km]")
    ax3.set_xlabel("Energy_Loss[kt/km]")
    KE_diff = np.diff(KE)
    z_diff = np.diff(z)
    KE_unit = abs(KE_diff / z_diff) * 1e3 / 4.18E12

    #chec that the KE is in the right index location
    ax3.plot(KE_unit, z[:-1] / 1e3)
    # ax3.axhline(y=z[KE_unit.argmax()]/1e3, xmin=KE_unit[0], xmax=KE_unit[-1],color='r', linestyle='--',linewidth=0.5)
    z_burst = z[KE_unit.argmax()] / 1e3
    ax3.axhline(y=z_burst, color='r', linestyle='--', linewidth=1)
    ax3.axvline(x=KE_unit.max(), color='r', linestyle='--', linewidth=1)
    ax3.annotate('{:.2E}'.format(z_burst), xy=(0, 2 * z_burst), color='r')
    ax3.annotate('{:.2E}'.format(np.max(KE_unit)), xy=(KE_unit.max() + KE_unit.max() / 100, 8 * z_burst),
                 color='r', rotation=90)
    print(KE_unit[burst_index], z[burst_index])
    ax3.annotate('*', xy=(KE_unit[burst_index], z[burst_index] / 1e3))
    # z_ana_diff,KE_ana_diff=plot()
    # ax3.plot(z_ana_diff,KE_ana_diff)# need to lay analytical solution on top!
    plt.show()


def ode_solver_pre_burst(t, state):
    f = np.zeros_like(state)
    v, m, theta, z, x, r = state
    f[0] = dv(z, r, v, theta, m)
    f[1] = dm(z, r, v)
    f[2] = dtheta(theta, v, z, m, r)
    f[3] = dz(theta, v)
    f[4] = dx(theta, v, z)
    f[5] = 0
    return f


def ode_solver_post_burst(t, state):
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
    t0=0
    tmax=40.
    dt = 0.1
    t = np.arange(t0, tmax, dt)

    states = solve_ivp(ode_solver_pre_burst, (0, 1.1 * tmax), state0, t_eval=t, method='RK45')

    # need to find index where breakup occurs

    v = np.array(states.y[0])
    z = np.array(states.y[3])
    tensile_stress = rho_a(z) * v ** 2
    burst_index = np.argmax(tensile_stress > Y)  # need to have a case for if yield strength is not exceeded
    if burst_index == 0:
        print('Cratering Event')
    else:
        print('Airburst Event')
        
    t_new = t[burst_index]
    t2 = np.arange(t_new, tmax, dt)
    state0 = states.y[:, burst_index]

    states_2 = solve_ivp(ode_solver_post_burst, (t_new, 1.1 * tmax), state0, t_eval=t2, method='RK45')

    solution = np.concatenate((states.y[:, 0:burst_index], states_2.y), axis=1)

    v=solution[0]
    m=solution[1]
    theta=solution[2]
    z=solution[3]
    x=solution[4]
    KE=0.5*v**2*m
    r=solution[5]

    plot(t, v, z, KE, burst_index)



main()

