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



def confidence_prediction(num):
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
    rho_m_mean = 3300
    rho_m_sigma = 200
    Y_mean = 2E6
    Y_sigma = 0.1E6


    n = 0
    random_array = np.zeros((8, num))

    while n < num:
        random_array[0, n] = np.random.normal(velocity_mean, velocity_sigma, 1)  # velocity
        random_array[1, n] = np.random.normal(mass_mean, mass_sigma, 1)  # mass
        random_array[2, n] = np.random.normal(theta_mean, theta_sigma, 1)  # theta
        random_array[3, n] = np.random.normal(altitude_mean, altitude_sigma, 1)  # altitude
        random_array[4, n] = np.random.normal(horizontal_mean, horizontal_sigma, 1)  # horizontal
        random_array[5, n] = np.random.normal(radius_mean, radius_sigma, 1)  # radius
        random_array[6, n] = np.random.normal(rho_m_mean, rho_m_sigma, 1)  # radius
        random_array[7, n] = np.random.normal(Y_mean, Y_sigma, 1)  # radius



        n += 1
        
    return random_array

def find_ke_max(data):
    t, v,m,theta, z,x, KE,r, burst_index, airburst_event = data
    
    z_diff = np.diff(z)
    z_diff = np.append(z_diff,z_diff[-1])
    


    KE_km_kT = np.diff(KE)/np.diff(z/1000)/4.184e12
    KE_km_kT = np.append(KE_km_kT,KE_km_kT[-1])

    ke_max_value = KE_km_kT.max()
    ke_max_height = z[np.argmax(KE_km_kT == ke_max_value)]
    return ke_max_value, ke_max_height


def ensemble_distribution():
    random_array = confidence_prediction()
#    print(random_array[:,0])
    for i in range(len(random_array[0,:])):
        er.initial_state =random_array[:,i]
        