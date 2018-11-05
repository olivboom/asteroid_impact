# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 15:15:37 2018

@author: Laura Su
"""

import numpy as np
import matplotlib.pyplot as plt


def read_alt_density(filename):
    file = open(filename, "r")

    alt = []
    atm_dens = []
    scale_hei = []

    for line in file:
        line = line.strip()
        if line.startswith("#"):
            continue
        altitude, dens, height = line.split()
        alt.append(float(altitude))
        atm_dens.append(float(dens))
        scale_hei.append(float(height))

    return alt, atm_dens, scale_hei


def sqr_error(p, xi, yi):
    diff2 = (p(xi)- yi)**2
    return diff2.sum()


if __name__ == '__main__':
    alt, dens, height = read_alt_density("AltitudeDensityTable.csv")
    dens = np.array(dens)

    alt_div = np.array([alt[i]/height[i] for i in range(len(alt))])

    log_dens = np.log(dens)


    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    
    plt.plot(alt_div, dens,  ".")
    
    ax1.plot(alt_div, log_dens, "r*")

    
    poly_coeffs = np.polyfit(alt_div, log_dens, 1)
    p1 = np.poly1d(poly_coeffs)
    print("Try".format(poly_coeffs))
    
    x = np.linspace(min(alt_div), max(alt_div))
    ax1.plot(x,p1(x),"b")
    
    ss_res = sqr_error(p1, alt_div, log_dens)
    ss_tot = np.sum((np.mean(log_dens) - log_dens)**2)
    r2 = 1. - ss_res/ss_tot
    print("R2 =", r2)