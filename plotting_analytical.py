# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 09:30:14 2018

@author: Hamed
"""

import initialisation 

import numpy as np
import matplotlib.pyplot as plt



def initialise_parameters(C_D=1, H=8000, rho_0=1.2):
    """
    Setting the parameters and variables used in the analytical solution
    to create the plot
    Returns
    -----------------
    An array containing the following quantities
    C_D   : float,  dimensionless
            Coefficient of drag.
    m     : float, kg
            Mass of asteroid
            Atmospheric scale height
    rho_0 : float, kg m^-3
            Sea level atmospheric density
    """


    print("here")

    return C_D, H, rho_0

def initialise_variables(v_init =19e3,m_init = 12e6, theta_init=20, z_init =100e3, diam_init = 19.5):
    A = np.pi * (diam_init/2)**2
    Theta = theta_init * (np.pi)/180
    
    
    return A, Theta, z_init, v_init, m_init
#
C_D, H, rho_0 = initialise_parameters()
A, Theta, z_init, v_init, m_init = initialise_variables()


def analytical():
    
    num = 100
    z = np.linspace(z_init,0, num)
    
    
    alpha = (H * A * rho_0)/(2*m_init*np.sin(Theta))
    beta = np.exp(-z_init/H)
    
    
    v = v_init * np.exp( alpha*beta - alpha * np.exp(-z/H))
    
    
    
    
    v_diff = np.diff(v)
    v_diff = np.append(v_diff, v_diff[-1])
    
    z_diff = np.diff(z)
    z_diff = np.append(z_diff,z_diff[-1])
    
    _1kT = 4.184E12
    
    KE = (1/2 * m_init *v**2)
    KE_diff = np.diff(KE)
    KE_diff = np.append(KE_diff,KE_diff[-1])
    
    KE_unit = abs(KE_diff/z_diff) * 1e3/_1kT
#    print(np.max(KE_unit))
#    
#    print(KE)
    
    x = np.abs(KE_diff/(z_diff/1000)/_1kT)
    y = z/1000
    plt.plot(np.abs(KE_diff/(z_diff/1000)) /_1kT, z/1000)
    plt.ylabel("Height above ground (m)")
    plt.xlabel("KE (kT km^1)")
    plt.show()
    
    return x,y
    
#if __name__ == "__main__":
#    analytical()
