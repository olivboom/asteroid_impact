# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 09:30:14 2018

@author: Hamed
"""
import numpy as np
import matplotlib.pyplot as plt

def initial_parameter():
    """
    Setting the parameters used in the analytical solution
    to create the plot

    Returns
    -----------------
    An array containing the following quantities

    C_D   : float,  dimensionless
            Coefficient of drag.

    m     : float, kg
            Mass of asteroid

    A     : float, m^2
            Projected surface area of asteroid

    Theta : float, radians
            Angle of incidence into the atmosphere

    H     : float, m
            Atmospheric scale height

    rho_0 : float, kg m^-3
            Sea level atmospheric density

    z_init : float, m
             Asteroid starting height
             
    v_init : float ms^-1
             Asteroid starting velocity

    """
    global C_D
    global m
    global A
    global Theta
    global H
    global rho_0
    global z_init
    global v_init

    print("here")

    C_D = 1
    m = 1e4
    A = 10
    Theta = 30 * (np.pi)/180
    H = 8000
    rho_0 = 1.2
    z_init = 1e5
    v_init = 2000




def plot():
    """ Plots analytical solution for simple case specified in project notes"""
    initial_parameter()
    num = 2000000
    z = np.linspace(0, z_init, num)
    step = (z_init/num )/1000
    print(step)

    const = -(rho_0 * A * H)/(2*m*np.sin(Theta))
    ln_v = const * np.exp(-z/H) - const* np.exp(-z_init/H) + np.log(v_init)

    v = np.exp(ln_v)
    v_diff = np.diff(v)
    
    z_diff = np.diff(z)
    
    
    _1kT = 4.184E12
    
    KE = (1/2 * m *v**2)
    KE_kT = KE/_1kT
    KE_diff = np.diff(KE_kT)/step
    KE_diff = np.append(KE_diff, 0)
    
    
    
#    KE_lost_kT
    
    plt.plot(KE_diff, z)
#    plt.xlim(, 0)
    plt.ylabel("Height above ground (m)")
    plt.xlabel("KE (kT km^-1)")
    
    return z_diff

print(plot())

plt.figure()
plot()
