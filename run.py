import initialisation
import eroscode
#import plotting_analytical
import matplotlib.pyplot as plt
import numpy as np


def settolerance(tol):
    eroscode.tol = tol

#set parameters
def run(tol):
    initialisation.set_parameters("Earth", analytical=True)
    initialisation.set_variables("Chelyabinsk")
    settolerance(tol)
    return eroscode.main()


def run_pre(tol):
    ## without burst
    initialisation.set_parameters("Earth", analytical=True)
    initialisation.set_variables("Chelyabinsk")
    settolerance(tol)
    return eroscode.main_pre()

def run_radau():
    initialisation.set_parameters("Earth", analytical=True)
    initialisation.set_variables("Chelyabinsk")
    return eroscode.main_radau()

def run_radau_pre():
    initialisation.set_parameters("Earth", analytical=True)
    initialisation.set_variables("Chelyabinsk")
    return eroscode.main_radau_pre()


#data = run(1E-5)


#unitke,z = find_ke_max(data)


#include assert to prevent misspelling

#print("done")
#
#analytical_numerical = eroscode.final_state
#
#t, v,m,theta, z_1,x, KE_1,r, burst_index, airburst_event = analytical_numerical
##np.abs(KE_diff/(z_diffp/1000)) /_1kT, z/1000
#plt.figure()
#
#KE_2, z_2 = plotting_analytical.plot()
#
##plt.plot(KE_,z_)
#
#def plot(z, KE):
#    
#    v_diff = np.diff(v)
#    v_diff = np.append(v_diff, v_diff[-1])
#    
#    z_diff = np.diff(z)
#    z_diff = np.append(z_diff,z_diff[-1])
#    
#    _1kT = 4.184E12
#    
#    KE = (1/2 * m *v**2)
#    KE_diff = np.diff(KE)
#    KE_diff = np.append(KE_diff,KE_diff[-1])
#    
#    KE_unit = abs(KE_diff/z_diff) * 1e3/_1kT
#    print(np.max(KE_unit))
#    
#    print(KE)
#    
#        
#    plt.plot(np.abs(KE_diff/(z_diff/1000)) /_1kT, z/1000)
#    plt.ylabel("Height above ground (m)")
#    plt.xlabel("KE (kT km^1)")
#    plt.ylim(bottom = 0)
#    plt.show()
#
##
##    plt.show()
#plot(z_2,KE_2)
##a, b = plotting_analytical.plot()
#
