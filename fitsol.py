# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 15:15:37 2018
@author: Laura Su
"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy.interpolate import CubicSpline
from scipy import interpolate
import scipy.optimize as optimization
import run
import numpy.polynomial.polynomial as poly
import eroscode
#import plotting_analytical_assumptions


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

'''
def error_tol(sol_ana, sol_num):
    sol_ana = np.array(sol_ana)
    sol_num = np.array(sol_num)
    err = abs(sol_ana - sol_num)
    
    return err

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


def error_tol_th(sol_ana, sol_num):
    sol_ana = np.array(sol_ana)
    sol_num = np.array(sol_num)
    err = abs(sol_ana[-1] - sol_num[-1])
    return err
'''

#global Y
#Y = 2e6

def fitsol():
    z_ana, KE_ana = read_chelyabinsk("data\ChelyabinskEnergyAltitude.csv")
    KE_ana = np.array(KE_ana)
    z_ana = np.array(z_ana)
    #plt.plot(stats.linregress(z_ana, KE_ana))
    #np.interp(z_ana, KE_ana)
    #[np.polyfit(z_ana[i:(i+2)], KE_ana[i:(i+2)],1) for i in range(len(z_ana)-1)]
    #z=np.polyfit(z_ana,KE_ana,2)
    #p = np.poly1d(z)
    KEmaxind=KE_ana.argmax()
    weights=np.ones(len(z_ana))
    weights[KEmaxind-3:KEmaxind+3]=4
    coefs = poly.polyfit(z_ana, KE_ana, 10,w=weights)
    #Y=run.setstrength(0.5E6)
    Y=0.5E6
    R=8.75
    badfit=1
    LSE=[]
    deg20=eroscode.deg_to_rad(20)
    i=0
    LSE=[1E3,0.9E3]
    Rlist=[0,0]
    Ylist=[0,0]
    i=0
    Rinc=True
    Yinc=False
    #for i in range(500):
    #while (not np.isclose(LSE[-1],LSE[-2],rtol=1e-03, atol=1e-05)) or i>500:
    while(i<160):
        i+=1
        R=R+0.1
        Y=Y+0.05E6
        final_state=run.run_custom(19E3,12E6,deg20,1E5,0,R,Y,planet="Earth",show =False)
        z=final_state[4]/1E3
        KE=final_state[6]/4.18E12
        mask=np.logical_and(z<=max(z_ana), z>=min(z_ana))
        z_num=z[mask][:-1]
        KE_unit=np.diff(KE)/np.diff(z)
        KE_num=KE_unit[mask[:-1]]
        KE_fit = poly.polyval(z_num, coefs)
        LSE_val=np.square(np.subtract(KE_num, KE_fit)).mean()
        #while(LSE[-1]>LSE[-2]):
        LSE.append(LSE_val)
        Ylist.append(Y)
        Rlist.append(R)
        print(LSE_val, R, Y)
        #if LSE[-1]<LSE[-2]:
        #    break
        #Y+=0.05E6
        i+=1
    Y=Ylist[LSE.index(min(LSE))]
    R=Rlist[LSE.index(min(LSE))]
    print(Y,R)
    plt.figure()
    plt.plot(z_ana,KE_ana, label='Chelyabinsk Table Data',color='b')
    plt.plot(z_num,KE_fit,label='Polyfit',color='r')
    plt.plot(z_num,KE_num,label='Numeric Solution',color='g')
    #plt.plot(p(z),label='polyfit')
    plt.xlabel('z[km]')
    plt.ylabel('dE/dz[kt/km]')
    plt.title('Chelyabinsk Fit')
    plt.legend()
    plt.show()
    return Y,R

fitsol()

'''
    for elt in tol_array:
        final_state = run.run(elt)
        
        z = final_state[4]/1000
        
        #mask_num = (z >= 30.4)
        
        KE_diff = np.diff(final_state[6])
        z_diff = np.diff(final_state[4])
        KE_unit = KE_diff/z_diff*1000/4.148E12
        KE_unit = np.append(KE_unit, 0)
        
        
        error = error_tol_th(KE_ana,KE_unit)
        err_num.append(error)
        
    return tol_array, err_num    
            

if __name__ == '__main__':
#    z, KE = read_chelyabinsk("ChelyabinskEnergyAltitude.csv")    
#    print(z, KE)
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    fig.tight_layout(w_pad=5, h_pad=5)

    tol_array, err = many_tol_out()
    tol_array1, err1 = many_tol()
    
    print(err1)
    
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
'''