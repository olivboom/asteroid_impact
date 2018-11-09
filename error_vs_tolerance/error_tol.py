# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 15:15:37 2018

@author: Laura Su
"""

import numpy as np
import matplotlib.pyplot as plt
import run


def error_tol(sol_ana, sol_num):
    sol_ana = np.array(sol_ana)
    sol_num = np.array(sol_num)
    err = abs(sol_ana - sol_num)

    return err


def tol_err_without():
    """
    Example without fragmentation

    :return: array of different tolerances and array of errors
    """
    tol_array = np.logspace(-5, -1, 10)
    err_num = []
    index = 150
    final_state_ana = run.run_radau_pre()
    KE_diff_ana = np.diff(final_state_ana[6])
    z_diff_ana = np.diff(final_state_ana[4])
    KE_ana = KE_diff_ana / z_diff_ana * 1000 / 4.148E12
    KE_ana = np.append(KE_ana, 0)

    for elt in tol_array:
        final_state = run.run_pre(elt)

        KE_diff = np.diff(final_state[6])
        z_diff = np.diff(final_state[4])
        KE_unit = KE_diff / z_diff * 1000 / 4.148E12
        KE_unit = np.append(KE_unit, 0)

        error = error_tol(KE_ana[index], KE_unit[index])
        err_num.append(error)

    return tol_array, err_num


def tol_err_with():
    """
    Example with fragmentation
    :return: array of different tolerances, array of errors
    """
    tol_array = np.logspace(-5, -1, 10)
    err_num = []
    index = 175

    final_state_ana = run.run_radau()

    KE_diff_ana = np.diff(final_state_ana[6])
    z_diff_ana = np.diff(final_state_ana[4])
    KE_ana = KE_diff_ana / z_diff_ana * 1000 / 4.148E12
    KE_ana = np.append(KE_ana, 0)

    for elt in tol_array:
        final_state = run.run(elt)

        KE_diff = np.diff(final_state[6])
        z_diff = np.diff(final_state[4])
        KE_unit = KE_diff / z_diff * 1000 / 4.148E12
        KE_unit = np.append(KE_unit, 0)

        error = error_tol(KE_ana[index], KE_unit[index])
        err_num.append(error)

    return tol_array, err_num


if __name__ == '__main__':
    # Plotting the two examples, error vs tolerance

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    fig.tight_layout(w_pad=5, h_pad=5)

    tol_array, err = tol_err_without()
    tol_array1, err1 = tol_err_with()
    print("without fragmentation", err)
    print("with fragmentation", err1)

    axs[0].loglog(tol_array, err, "g", label="example without fragmentation")
    axs[1].loglog(tol_array1, err1, "g", label="example with fragmentation")

    axs[0].set_title("Error vs tolerance, without fragmentation", fontsize=16)
    axs[0].set_xlabel("tolerance", fontsize=12)
    axs[0].set_ylabel("error", fontsize=12)
    axs[0].legend(loc="best")

    axs[1].set_title("Error vs tolerance, with fragmentation", fontsize=16)
    axs[1].set_xlabel("tolerance", fontsize=12)
    axs[1].set_ylabel("error", fontsize=12)
    axs[1].legend(loc="best")

    plt.show()
