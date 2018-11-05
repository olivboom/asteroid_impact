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
    r = 3
    v = 20e3
    rho_asteroid = 3300
    Vol = Volume(r)
    m = Vol * rho_asteroid
    theta_initial = 18.3 * np.pi /180
    z = 100e3

def KE(m,v):
    return 0.5 * m * v**2

def dv(z, r):
    return ((-C_D * rho_a(z) * A(r) * v**3)/(2*m) + g * np.sin(theta)) 
    

def rho_a(z):
    return rho_0 * np.exp(-z/H)

def dz(theta,v):
    return -v*np.sin(theta)

def dx(theta, v, z):
    return (v*np.cos(theta))/(1 + z/R_E)

def dm(z,v):
    return (-C_H * rho_a(z) * A(r) * v**3)/(2*Q)

def A(r):
    return np.pi * r**2

def Volume(r):
    return 4/3 * np.pi * r**3
    


def main():
    initial()
    max_iter:
    
    
    while n < max_iter:
        
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
    
    
    
    

    
    

if __name__ == "main":
    main()