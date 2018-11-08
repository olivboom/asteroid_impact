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
    global z_init_m


    print("here")

    C_D = 1    
    m = 12e6
    A = np.pi * (19.5/2)**2
    Theta = 20 * (np.pi)/180
    H = 8000
    rho_0 = 1.2
    z_init = 100e3
    v_init = 19e3




def plot(z):
    initial_parameter()
    #num = 100
    #z = np.linspace(z_init,0, num)
    
    
    
    #step = (z_init/num )
    #print(step)
    
    print("H: ", H)
    print("A: ", A)
    print("rho_0: ", rho_0)
    print("m: ", m)
    print("Theta: ", Theta)
    
    
    alpha = (H * A * rho_0)/(2*m*np.sin(Theta))
    beta = np.exp(-z_init/H)
    
    
    v = v_init * np.exp( alpha*beta - alpha * np.exp(-z/H))
    
    
    
    
    v_diff = np.diff(v)
    v_diff = np.append(v_diff, v_diff[-1])
    
    z_diff = np.diff(z)
    z_diff = np.append(z_diff,z_diff[-1])
    
    _1kT = 4.184E12
    
    KE = (1/2 * m *v**2)
    KE_diff = np.diff(KE)
    KE_diff = np.append(KE_diff,KE_diff[-1])
    
    KE_unit = abs(KE_diff/z_diff) * 1e3/_1kT
    print(np.max(KE_unit))
    
    print(KE_unit)
    
        
    #plt.plot(np.abs(KE_diff/(z_diff/1000)) /_1kT, z/1000)
    #plt.ylabel("Height above ground (m)")
    #plt.xlabel("KE (kT km^1)")
    #plt.show()
    
    
    return KE_unit, v
#plot()
