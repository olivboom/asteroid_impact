# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 09:30:14 2018

@author: Hamed
"""
import numpy as np
import matplotlib.pyplot as plt

def initial_parameter():
    """
    Creating the parameters used in the analytical solution
    to create the plot

    Returns
    -----------------
    An array containing the following quantities

    C_D  : float,  dimensionless
        Coefficient of drag.
        
    m  : float, kg
         Mass
         
    A  : float, m^2
         Area
         
    Theta: float, radians
           Angle of incidence
           
    H   : float, m
          Atmospheric scale height        
    """
    global C_D
    global m
    global A
    global Theta
    global H
    
    C_D = 1
    m = 1e3
    A = 4
    Theta = 15 * np.sin(np.pi)/180
    H = 8000
    
#    return np.array((C_D,
#                    m,
#                    A,
#                    Theta,
#                    H))
    
    
    
def plot():
    initial_parameter()

    z = np.linspace(0,1e4, 100)
    
    ln_v = ((-H*C_D*A)/(2*m*np.sin(Theta)))*np.exp(-z/H)
    print(ln_v)
    
    v = np.exp(ln_v)
    
    plt.plot(z,ln_v)
    plt.xlabel("Height above ground (m)")
    plt.ylabel("Velocity (m/s)")
    return z,ln_v
    
    
plot()
    
    
    