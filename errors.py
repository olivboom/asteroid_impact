# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 19:17:05 2018

@author: Laura Su
"""
import numpy as np
import matplotlib.pyplot as plt
import eroscode_assumptions
import plotting_analytical_assumptions
import eroscode_IE_assumptions
#error vs time step size

############################################################
##to be modified... eventually
###just get z and v!
z, v_num, KE_unit = eroscode_assumptions.main()
KE_unit_ana, v_ana = plotting_analytical_assumptions.plot(z)

z_IE, v_IE = eroscode_IE_assumptions.main()
KE_meh, v_ana_IE = plotting_analytical_assumptions.plot(np.array(z_IE))
############################################################


def error(sol_ana, sol_num):
    sol_ana = np.array(sol_ana)
    sol_num = np.array(sol_num)
    err = abs(sol_ana - sol_num)
    
    return err


z = np.array(z)
mask = (z >= 0)

z_pos = z[mask]
v_num_pos = v_num[mask]
v_ana_pos = v_ana[mask]


print(np.max(error(v_ana_pos, v_num_pos)))

fig, axs = plt.subplots(2, 1, figsize=(8, 6))
fig.tight_layout(w_pad=5, h_pad=5)


axs[0].plot(z_pos/1000, v_ana_pos/1000, "g", label = "Speed, analytical") 
axs[0].plot(z_pos/1000, v_num_pos/1000, "b", label = "Speed, numerical")

axs[1].plot(z_pos/1000, error(v_ana_pos/1000,v_num_pos/1000), "r", label = "Error")

axs[0].set_title('Numerical vs analytical solution', fontsize=16)
axs[0].set_xlabel("Height above ground (km)", fontsize=12)
axs[0].set_ylabel("Speed (km/s)", fontsize=12)
axs[0].legend(loc ="best")

axs[1].set_title('Error between the two solutions', fontsize=16)
axs[1].set_xlabel("Height above ground (km)", fontsize=12)
axs[1].set_ylabel("Speed (km/s)", fontsize=12)
axs[1].legend(loc ="best")


z_IE = np.array(z_IE)
v_IE = np.array(v_IE)
v_ana_IE = np.array(v_ana_IE)

mask = (z_IE >= 0)


z_IE_pos = z_IE[mask]
v_num_IE_pos = v_IE[mask]
v_ana_IE_pos = v_ana_IE[mask]


fig2, ax = plt.subplots(2, 1, figsize=(8, 6))
fig2.tight_layout(w_pad=5, h_pad=5)


ax[0].plot(z_IE_pos/1000, v_ana_IE_pos/1000, "g", label = "Speed, analytical") 
ax[0].plot(z_IE_pos/1000, v_num_IE_pos/1000, "b", label = "Speed, numerical (Improved Euler)") 

ax[1].plot(z_IE_pos/1000, error(v_ana_IE_pos/1000,v_num_IE_pos/1000), "r", label = "Error")

ax[0].set_title('Numerical (Improved Euler) vs analytical solution', fontsize=16)
ax[0].set_xlabel("Height above ground (km)", fontsize=12)
ax[0].set_ylabel("Speed (km/s)", fontsize=12)
ax[0].legend(loc ="best")

ax[1].set_title('Error between the two solutions', fontsize=16)
ax[1].set_xlabel("Height above ground (km)", fontsize=12)
ax[1].set_ylabel("Speed (km/s)", fontsize=12)
ax[1].legend(loc ="best")

plt.show()