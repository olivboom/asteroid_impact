"""

want to set boundary conditions for each parameter and variables
then have an extremely large array which is allocated the parameters
or variables in accordance to the statistical properties given
then run a loop that outputs all of the various outputs and plot
is as a statistical ensemble

Need to be able to calculate the uncertainty in altitude and magnitude of peak energy release
"""

# import scipy.stats as stat
import numpy as np
import eroscode as er


def confidence_prediction():
    """
    This statistical analysis if for scenarios where the initial
    conditions of the asteroid are known but there is a lack of
    confidence in the initial conditions and so a gaussian
    distribution is used to have a confidence as to the likely
    outcome of the scenario
    :return:
    """

    # confidence_level = 0.1

    velocity_mean = 19e3
    velocity_sigma = 1e3  # velocity_mean * confidence_level
    mass_mean = 12e6
    mass_sigma = 1e3
    theta_mean = er.deg_to_rad(20)
    theta_sigma = er.deg_to_rad(2)
    altitude_mean = 100e3
    altitude_sigma = 1e3
    horizontal_mean = 0
    horizontal_sigma = 0
    radius_mean = 10
    radius_sigma = 1

    n = 0
    random_array = np.zeros((6, 10))

    while n < 10:
        random_array[0, n] = np.random.normal(velocity_mean, velocity_sigma, 1)  # velocity
        random_array[1, n] = np.random.normal(mass_mean, mass_sigma, 1)  # mass
        random_array[2, n] = np.random.normal(theta_mean, theta_sigma, 1)  # theta
        random_array[3, n] = np.random.normal(altitude_mean, altitude_sigma, 1)  # altitude
        random_array[4, n] = np.random.normal(horizontal_mean, horizontal_sigma, 1)  # horizontal
        random_array[5, n] = np.random.normal(radius_mean, radius_sigma, 1)  # radius

        n += 1


confidence_prediction()
