# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 16:48:56 2018

@author: Hamed
"""

import eroscode
import numpy as np

def array_variables():
    Tunguska = np.array([20e3, 11e6, eroscode.deg_to_rad(45), 100e3, 0, 25 ])
    Analytical = np.array([19e3, 11e6, eroscode.deg_to_rad(45), 100e3, 0, 25 ])
    Chelyabinsk = np.array([19e3, 12e6, eroscode.deg_to_rad(20), 100e3, 0, 9.75 ])
    
    
    return Tunguska, Analytical, Chelyabinsk

def set_parameters(planet="Earth"):
    eroscode.C_D = 1
    eroscode.C_H = 0.1
    eroscode.Q = 1E7
    eroscode.C_L = 1E-3
    eroscode.alpha = 0.3
    eroscode.rho_0 = 1.2
    
    
    if planet == "Earth":
        eroscode.r = 6.371E6
        eroscode.g_E = 9.81
        eroscode.H = 8000

        
    elif planet == "Mars":
        eroscode.r = 3390e3
        eroscode.g_E = 3.8
        eroscode.H = 10800


    

def set_variables(name="Tunguska"):
    
    
    v_init = None
    m_init = None
    theta_init = None
    z_init = None
    x_init = None
    r_init = None

    variables_values = [v_init, m_init, theta_init, 
                      z_init, x_init, r_init]
    
    variables_names = ["v_init", "m_init", "theta_init", 
                      "z_init", "x_init", "r_init"]

    
    Tunguska, Analytical, Chelyabinsk = array_variables()

    if name == "Tunguska":
        print("T")
        variables_values = Tunguska
        
    if name == "Analytical":
        print("a")
        variables_values = Analytical
        
    if name == "Chelyabinsk":
        print("C")
        variables_values = Chelyabinsk

    for i, variable in enumerate(variables_names):
        variables_names[i] = "eroscode." + variable
        
    for i in range(len(variables_values)):
        exec(str(variables_names[i]) + "=" + str(variables_values[i]))
        
    return np.array([v_init,
                 m_init,
                 theta_init,
                 z_init,
                 x_init,
                 r_init])



