# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 16:48:56 2018

@author: Hamed
"""

import eroscode
import numpy as np
import statistical_ensemble as se
import plotting_analytical

def deg_to_rad(deg):
    return deg*np.pi/ 180


def rad_to_degrees(rad):
    return rad*180/np.pi


def asteroid_data():
    #ASTEROID = m_init,v_init, theta, z_init, x_init, r_init, rho_m, Y
    
    Tunguska = np.array([20e3, 11e6, deg_to_rad(45), 100e3, 0, 25, 3300, 2E6 ])
    Analytical = np.array([19e3, 12e6, deg_to_rad(20), 100e3, 0, 9.75, 3300,1E6 ])
    Chelyabinsk = np.array([19e3, 12e6, deg_to_rad(20), 100e3, 0, 9.75, 3300,1E6 ])
    
    return Tunguska, Analytical, Chelyabinsk

def settolerance(tol):
    eroscode.tol = tol

def set_parameters(planet="Earth", analytical_assumption=False):
    print("analytical: ", analytical_assumption)
    
    eroscode.analytical = analytical_assumption
    eroscode.C_D = 1
    eroscode.C_H = 0.1
    eroscode.Q = 1E7
    eroscode.C_L = 1E-3
    eroscode.alpha = 0.3
    eroscode.tol = 1e-10
    
        
    
    
    if planet == "Earth":
        eroscode.R_E = 6.371E6
        eroscode.g_E = 9.81
        eroscode.H = 8000
        eroscode.rho_0 = 1.2


        
    elif planet == "Mars":
        eroscode.R_E = 3396e3
        eroscode.g_E =9.81 * 0.38
        eroscode.H = 11.1e3
        eroscode.rho_0 = 0.02 #find out true value

    if analytical_assumption == True:
        eroscode.g_E = 0
        eroscode.R_E = np.inf
        eroscode.C_L = 0

def set_parameters_analytical_solution(planet="Earth"):
    plotting_analytical.C_D = 1
    plotting_analytical.C_H = 0.1
    plotting_analytical.Q = 1E7
    plotting_analytical.alpha = 0.3
    plotting_analytical.g_E = 0
    plotting_analytical.R_E = np.inf
    plotting_analytical.C_L = 0

    
    
    if planet == "Earth":
        plotting_analytical.R_E = 6.371E6
        plotting_analytical.H = 8000
        plotting_analytical.rho_0 = 1.2


        
    elif planet == "Mars":
        plotting_analytical.R_E = 3390e3
        plotting_analytical.H = 10800
        plotting_analytical.rho_0 = 1.2 #find out true value
        
    


set_parameters_analytical_solution(planet="Earth")




def set_variables(name="Tunguska"):
    
    
    v_init = None
    m_init = None
    theta_init = None
    z_init = None
    x_init = None
    r_init = None
    rho_m = None
    Y = None
    
    state_0 = [v_init, m_init, theta_init, 
                      z_init, x_init, r_init, rho_m, Y]
    Tunguska, Analytical, Chelyabinsk = asteroid_data()

#
    if name == "Tunguska":
#        print("T")
        state_0 = Tunguska
        
    if name == "Analytical Assumptions":
#        print("a")
        state_0 = Analytical
        
    if name == "Chelyabinsk":
#        print("C")
        state_0 = Chelyabinsk
    
    if name == "Ensemble":
        state_0 = se.confidence_prediction()
        
        
    
    return state_0
#
    
def set_variables_custom(variables):
    eroscode.initial_state = variables

def set_variables_ensemble(variables):
    eroscode.initial_state = variables
    
    
#set_variables()


