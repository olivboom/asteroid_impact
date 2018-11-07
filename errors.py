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

def vvvv(name="Tunguska"):
    
    
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

vvvv('Tunguska')
print(eroscode.theta_init)

vvvv('Chelyabinsk')
print(eroscode.theta_init)

vvvv('Analytical')
print(eroscode.theta_init)

#    
#    print(Tunguska_)
#    
#    for i, variable in enumerate(list_variables):
#        list_variables[i] = Tunguska_[i]
#    return list_variables
#    list_variables2 = ["v_init", "m_init", "theta_init", 
#                      "z_init", "x_init", "r_init"]
#    
#    
#    for i, variable in enumerate(list_variables):
#        print("here")
#        list_variables[i] = "eroscode." + variable
#        
#    Tunguska = "2"
#    
#    variable = name
#    execstr = "name = {}".format(variable)
#    exec(execstr)
#    print(execstr)
##    print(exec("Tungus"))
##    Tunguska_, Analytical_, Chelyabinsk_ = array_variables()
##    print(Tunguska_)
##    print(exec(name + "_"))
##    print(Tunguska)
###    print(Tung_data)
###    print(str(Tunguska))
##    print(type(exec(name)))
##    print(name)
###    print(Tunguska)
##    foo = "bar"
##    exec(foo + " = 'something else'")
##    print (bar)
#
#
#    
##    print(name_)
##    exec("name_" + "=name" )
##    print(name)
##    for i in range(len(name_)):
##        exec(list_variables[i] + "=" + str(name[i]))
    
#vvvv()

    
#




    
#def initial_variables():
#    """
#    Creating the initial conditions for the quantities
#    that will be solved in the ODE solver
#
#    Returns
#    -----------------
#    An array containing the following quantities
#
#    v_init : float
#    The inital velocity
#
#    m_init : float
#        The intial mass
#
#    theta_init : float
#        The initial entry angle
#
#    z_init : float
#        The initial altitude that the asteroid is measured from
#
#    x_init : float
#        The initial condition zeroing the horizontal displacement ******** In which axes
#
#    r_init : float
#        The initial condition for the radius of the asteroid
#    """
#
#    v_init = 19e3
#    m_init = 12e6
#    theta_init = deg_to_rad(20)
#    z_init = 100e3
#    x_init = 0
#    r_init = 19.5/2
#    return np.array([v_init,
#                     m_init,
#                     theta_init,
#                     z_init,
#                     x_init,
#                     r_init])
#
#
#
#def array_parameters():
#    """C_D, C_H, Q, C_L, alpha, H, rho_0, r, g_E"""
#    
#    Tumbuska = np.array()
#    Analytical = np.array()
#    Chelyabisnk = np.array([1, 0.1, 1E7, 1E-3, 0.3, 8000, 1.2, 6.371E6, 9.81])
#    
#    array = np.vstack(Tumbuska, Analytical, Chelyabinsk)
