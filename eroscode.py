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

global initial_state
global analytical
#global final_state

C_D = None
C_H = None
Q = None
C_L = None
alpha = None
H = None
rho_0 = None
R_E = None
g_E = None
rho_m=3.3E3
Y = 2e6


initial_state = None
analytical = False

#final_state = None


    


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
#    print(theta, v, z, m, r, R_E)
    return (g_E * np.cos(theta) / v) - ((C_L * rho_a(z) * area(r) * v) /
                                        (2 * m)) - ((v * np.cos(theta)) / (R_E + z))


def dx(theta, v, z):
#    print("//////", theta, v,z, R_E)
    return (v * np.cos(theta)) / (1 + z / R_E)


def dm(z, r, v):
    return (-C_H * rho_a(z) * area(r) * v ** 3) / (2 * Q)


def area(r):
    return np.pi * r ** 2


def volume(r, h):
    return np.pi*r**2*h


def dr(v, z, m, r):
    h=geth(r,m)
    return np.sqrt(7 / 2 * alpha * rho_a(z) / (m / volume(r, h))) * v


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
    ax3.annotate('*', xy=(KE_unit[burst_index], z[burst_index] / 1e3))
    # z_ana_diff,KE_ana_diff=plot()
    # ax3.plot(z_ana_diff,KE_ana_diff)# need to lay analytical solution on top!
    plt.show()


def ode_solver_pre_burst(t, state):
    f = np.zeros_like(state)
    v, m, theta, z, x, r = state
    f[0] = dv(z, r, v, theta, m)
    f[1] = dm(z, r, v)
    if analytical == True:
        f[1] = 0


    f[2] = dtheta(theta, v, z, m, r)
    f[3] = dz(theta, v)
    f[4] = dx(theta, v, z)
    f[5] = 0
    return f

#def settings(*vars):
#    if x == None:
        
def ode_solver_post_burst(t, state):
    f = np.zeros_like(state)
    v, m, theta, z, x, r = state
    f[0] = dv(z, r, v, theta, m)
    f[1] = dm(z, r, v)
    if analytical == True:
        f[1] = 0

    f[2] = dtheta(theta, v, z, m, r)
    f[3] = dz(theta, v)
    f[4] = dx(theta, v, z)
    f[5] = dr(v, z, m, r)
    return f


def main():
#    errors.set_parameters()
#    initialisation.set_parameters("Earth")
#    initialisation.set_variables("Tunguska")
    
    state0 = initial_state
    print(state0)

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
    
    global final_state
    
    print("C_D: ", C_D)
    print("C_H: ", C_H)
    print("Q: ", Q)
    print("C_L: ", C_L)
    print("alpha: ", alpha)
    print("H: ", H)
    print("R: ", R_E)
    print("g:", g_E)
    
    print(analytical)

    
    final_state = t, v,m,theta, z,x, KE,r, burst_index


#if __name__ == "__main__":
#    a = main()
