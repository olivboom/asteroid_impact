# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 09:30:14 2018

@author: Hamed
"""

import initialisation 

import numpy as np
import matplotlib.pyplot as plt

global C_D
global m
global A
global Theta
global H
global rho_0

global variables
global z_init
global v_init


#print(H)

#H = None
#print("H: ", H)
def returnparameters():
    return H,A,rho_0, Theta


def analytical(variables):
    
    v_init, m_init, theta_init, z_init, x_init, r_init = variables
    
    num = 100
#    z = np.linspace(z_init,0, num)
    
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
    
    print(KE)
    
        
    plt.plot(np.abs(KE_diff/(z_diff/1000)) /_1kT, z/1000)
    plt.ylabel("Height above ground (m)")
    plt.xlabel("KE (kT km^1)")
    plt.show()
    
    return z, KE
    
#if __name__ == "__main__":
#    analytical()
