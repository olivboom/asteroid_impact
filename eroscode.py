import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

C_D = 1       # Drag coefficient
C_H = 0.1     # Heat transfer coefficient
Q = 1E7       # Heat of ablation
C_L = 1E-3    # Lift coefficient
alpha = 0.3   # Dispersion coefficient
H = 8000      # Atmoshperic scale height
rho_0 = 1.2   # Atmoshpheric density at sea level
R_E = 6.371e6 # Radius of Earth
r = 3         # Radius of asteroid
g_E = 9.81    # grav constant on Earth
    
def initial():
    x_init = 0
    r_init = 3
    v_init = 20e3
    rho_m_init = 3300
    V_init = Volume(r)
    m_init = V_init * rho_m_init
    theta_init = 18.3 * np.pi /180
    z_init = 100e3
    
    return np.array([v_init, m_init, theta_init, 
                     z_init, x_init, r_init, rho_m_init])

def KE(m,v):
    return 0.5 * m * v**2

def dv(z, r, theta):
    return ((-C_D * rho_a(z) * A(r) * v**3)/(2*m) + g * np.sin(theta)) 
    

def rho_a(z):
    return rho_0 * np.exp(-z/H)

def dz(theta,v):
    return -v*np.sin(theta)

def dtheta(theta, v, z, m, r):
    return ((g_E*np.cos(theta)/v) - ((C_L*rho_a(z)*A(r)*v)/(2*m)) - ((v*np.cos(theta))/(R_E + z)))
    
    
def dx(theta, v, z):
    return (v*np.cos(theta))/(1 + z/R_E)

def dm(z,v):
    return (-C_H * rho_a(z) * A(r) * v**3)/(2*Q)

def A(r):
    return np.pi * r**2

def Volume(r):
    return 4/3 * np.pi * r**3

def dr():
    return (7/2*alpha*rho_a(z)/rho_m )   

def f(t, state):
    
    f = np.zeros_like(state)
    v, m, theta, z, x, r, rho_m = state
    f[0] = dv(z,r)
    f[1] = dm(z,v)
    f[2] = dtheta(theta,v,z,m,r)
    
state0 = initial()

dt = 0.1
t = np.arange(0.0, 40.0, dt)

    
def main():
    initial()
    max_iter = 100
    
    
    
#    while n < max_iter:
        
# =============================================================================
# def f(t, y):
#     f = np.zeros_like(y)
# 
#     v, m, theta, z ,x = y
#     
#     f[0] = (-C_D * rho_a 
#     f[1] = x * (rho - z) - y
#     f[2] = x * y - beta * z
#     
# =============================================================================
    
    
    
    

    
    

#if __name__ == "main":
#    main()