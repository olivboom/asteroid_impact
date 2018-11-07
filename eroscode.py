import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.ticker as ticker
import matplotlib as mpl
import pylab as plb
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

    v_init = 19.2e3
    m_init = 12e6
    theta_init = deg_to_rad(18.3)
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


def dr(z, m, r,v ):
    return np.sqrt(7 / 2 * alpha * rho_a(z) / rho_m)*v

def density(m,r):
    return m/volume(r)	


def mass(rho,r):
    return rho*volume(r)

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
    f[5] = 0
    return f


def ode_solver_burst(t, state):
    f = np.zeros_like(state)
    v, m, theta, z, x, r = state
    f[0] = dv(z, r, v, theta, m)
    f[1] = dm(z, r, v)
    f[2] = dtheta(theta, v, z, m, r)
    f[3] = dz(theta, v)
    f[4] = dx(theta, v, z)
    f[5] = dr(z, m, r,v)
    return f
'''
def improved_euler(f, u0, t0, t_max, dt):
    u = np.array(u0)
    t = np.array(t0)
    u_all = [u0]
    t_all = [t0]
    #f[5] = dr(z, m, r)
    while t < t_max:
        ue = u + dt*f(t, u)  # euler guess
        u = u + 0.5*dt* ( f(t, u) + f(t + dt, ue) )
        u_all.append(u)
        t = t + dt
        t_all.append(t)
    return np.array(u_all), np.array(t_all)
'''

def efun(x, a, b, c):
    return a*np.exp(-b*x)+c

def plot(t,v,z,KE):
    fig = plt.figure(figsize=(8,15))
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.7)
    
    #fig1=plt.subplot(311)
    ax1 = fig.add_subplot(311)
    ax1.plot(t, np.array(z)/1e3)
    ax1.set_xlabel("Time[s]")
    ax1.set_ylabel("Altitude[km]")

    popt, pcov = curve_fit(efun, t, np.array(z)/1e3,p0=(1, 1e-6, 1))
    yy = efun(t, *popt)
    ax1.plot(t,yy,'--r',label='{:.2f}*exp(-{:.2f}x)+{:.2f}'.format(popt[0],popt[1],popt[2]))
    ax1.legend()
    
    ax2=fig.add_subplot(312)
    ax2.set_xlabel("Speed[km/s]")
    ax2.set_ylabel("Altitude[km]")
    ax2.plot(np.array(v)/1e3,np.array(z)/1e3)
    ax2.set_xlim(ax2.get_xlim()[::-1])
    
    ax3=fig.add_subplot(313)
    ax3.set_ylabel("Altitude[km]")
    ax3.set_xlabel("Energy_Loss[kt/km]")
    KE_diff=np.diff(KE)
    z_diff=np.diff(z)
    KE_unit=abs(KE_diff/z_diff)*1e3/4.18E12
    ax3.plot(KE_unit, np.array(z[:-1])/1e3)
    #ax3.axhline(y=z[KE_unit.argmax()]/1e3, xmin=KE_unit[0], xmax=KE_unit[-1],color='r', linestyle='--',linewidth=0.5)
    z_burst=z[KE_unit.argmax()]/1e3
    ax3.axhline(y=z_burst,color='r', linestyle='--',linewidth=1)
    ax3.axvline(x=KE_unit.max(),color='r', linestyle='--',linewidth=1)
    ax3.annotate('{:.2E}'.format(z_burst),xy=(0,z_burst+0.5),color='r')
    ax3.annotate('{:.2E}'.format(KE_unit.argmax()),xy=(KE_unit.max()+KE_unit.max()/100,z_burst+z_burst),color='r',rotation=90)
    #z_ana_diff,KE_ana_diff=plot()
    #ax3.plot(z_ana_diff,KE_ana_diff)# need to lay analytical solution on top!
    plt.show()
    
def main():
    inital_parameter()
    state = initial_variables()
    t0=0
    tmax=40.
    dt = 0.1
    t = np.arange(t0, tmax, dt)
    # =============================================================================
    #     SOLVE_IVP
    # =============================================================================
    i=0
    #states = solve_ivp(ode_solver, (0, 50), state0, t_eval=t, method='RK45')  # need to fix the f tomorrow
    v=[state[0]]
    m=[state[1]]
    theta=[state[2]]
    z=[state[3]]
    x=[state[4]]
    KE=[0.5*v[-1]**2*m[-1]]
    r=[state[5]]
    #h=[geth(r[-1],m[-1])]
    Y=2E6
    
    #while t[i]<t[-1]:
    while t[i]<t[-1]:
        if Y>rho_a(z[-1])*v[-1]**2:
            state_e = state + dt*ode_solver(t[i], state)  # euler guess
            state = state + 0.5*dt* (ode_solver(t[i], state) + ode_solver(t[i+1], state_e) )
            v.append(state[0])
            m.append(state[1])
            theta.append(state[2])
            z.append(state[3])
            x.append(state[4])
            KE.append(0.5 * m[-1] * v[-1] ** 2)
            r.append(state[5])
            #h.append(geth(r[-1],m[-1]))
        else:
            state_e = state + dt*ode_solver_burst(t[i], state)  # euler guess
            state = state + 0.5*dt* (ode_solver_burst(t[i], state) + ode_solver_burst(t[i+1], state_e) )
            v.append(state[0])
            m.append(state[1])
            theta.append(state[2])
            z.append(state[3])
            x.append(state[4])
            KE.append(0.5 * m[-1] * v[-1] ** 2)
            r.append(state[5])  
            #h.append(h[-1])
            #rho_m.append(m[-1]/volume(r[-1],h[-1]))
        i+=1
    '''
    v = states.y[0]
    m = states.y[1]
    theta = states.y[2]
    z = states.y[3]
    x = states.y[4]
    r = states.y[5]
	'''
    plot(t,v,z,KE)
    #KE = 0.5 * m * v ** 2

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
