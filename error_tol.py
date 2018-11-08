# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 15:15:37 2018

@author: Laura Su
"""

import numpy as np
import matplotlib.pyplot as plt
import run
#import plotting_analytical_assumptions

"""
def read_chelyabinsk(filename):
    file = open(filename, "r")
    z = []
    KE = []
    file.readline()
    for line in file:
        height, energy = line.split(",")
        z.append(float(height))
        KE.append(float(energy))

    return z, KE 

"""
def error_tol(sol_ana, sol_num):
    sol_ana = np.array(sol_ana)
    sol_num = np.array(sol_num)
    err = abs(sol_ana - sol_num)
    
    return err

"""
def many_tol_out():
    ### Example without fragmentation
    tol_array = np.logspace(-5,-2,10)
    err_num = []
    index = 200
    for elt in tol_array:
        final_state = run.run(elt)
        KE_ana, v_ana = plotting_analytical_assumptions.plot(final_state[4])
        
        KE_diff = np.diff(final_state[6])
        z_diff = np.diff(final_state[4])
        KE_unit = KE_diff/z_diff *1000/4.148E12
        KE_unit = np.append(KE_unit, 0)
        
        error = error_tol(np.array(KE_ana),KE_unit)
        err_num.append(error[index])
    
    return tol_array, err_num
"""

def many_tol_out():
    ### Example without fragmentation
    tol_array = np.logspace(-5,-1,10)
    err_num = []
    index = 200
    final_state_ana = run.run_radau_pre()
    KE_diff_ana = np.diff(final_state_ana[6])
    z_diff_ana = np.diff(final_state_ana[4])
    KE_ana = KE_diff_ana/z_diff_ana*1000/4.148E12
    KE_ana = np.append(KE_ana, 0)
    
    for elt in tol_array:
        
        final_state = run.run_pre(elt)
        
        KE_diff = np.diff(final_state[6])
        z_diff = np.diff(final_state[4])
        KE_unit = KE_diff/z_diff*1000/4.148E12
        KE_unit = np.append(KE_unit, 0)
        
        
        error = error_tol(KE_ana[index],KE_unit[index])
        err_num.append(error)
    
    return tol_array, err_num



def many_tol():
    ### Example with fragmentation
    tol_array = np.logspace(-5,-1,10)
    err_num = []
    index = 200
    
    final_state_ana = run.run_radau()
    
    KE_diff_ana = np.diff(final_state_ana[6])
    z_diff_ana = np.diff(final_state_ana[4])
    KE_ana = KE_diff_ana/z_diff_ana*1000/4.148E12
    KE_ana = np.append(KE_ana, 0)
        
    for elt in tol_array:
        
        final_state = run.run(elt)
        
        KE_diff = np.diff(final_state[6])
        z_diff = np.diff(final_state[4])
        KE_unit = KE_diff/z_diff*1000/4.148E12
        KE_unit = np.append(KE_unit, 0)
        
        
        error = error_tol(KE_ana[index],KE_unit[index])
        err_num.append(error)
        
    return tol_array, err_num    
            

if __name__ == '__main__':
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    fig.tight_layout(w_pad=5, h_pad=5)

    tol_array, err = many_tol_out()
    tol_array1, err1 = many_tol()
    
    
    axs[0].loglog(tol_array, err, "g", label="example without fragmentation")
    axs[1].loglog(tol_array1, err1, "g", label="example with fragmentation")
    
    axs[0].set_title("Error vs tolerance, without fragmentation", fontsize=16)
    axs[0].set_xlabel("tolerance", fontsize=12)
    axs[0].set_ylabel("error", fontsize=12)
    axs[0].legend(loc ="best")

    axs[1].set_title("Error vs tolerance, with fragmentation", fontsize=16)
    axs[1].set_xlabel("tolerance", fontsize=12)
    axs[1].set_ylabel("error", fontsize=12)
    axs[1].legend(loc ="best")
    plt.show()