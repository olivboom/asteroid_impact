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
    y_mean = 2E6
    y_sigma = 0.1E6

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
        random_array[7, n] = np.random.normal(y_mean, y_sigma, 1)  # radius

        n += 1

    return random_array


def find_ke_max(data):
    t, v, m, theta, z, x, ke, r, burst_index, airburst_event = data
    ke_km_kt = np.diff(ke) / np.diff(z / 1000) / 4.184e12
    ke_km_kt = np.append(ke_km_kt, ke_km_kt[-1])
    ke_max_value = ke_km_kt.max()
    ke_max_height = z[np.argmax(ke_km_kt == ke_max_value)]

    return ke_max_value, ke_max_height


def ensemble_distribution():
    random_array = confidence_prediction(num=100)
    #    print(random_array[:,0])
    for i in range(len(random_array[0, :])):
        er.initial_state = random_array[:, i]
