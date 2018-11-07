'''
Whether the scenario is an airburst or a cratering event
- Give extremely strong astroid material and extremely weak
material and see if the airburst/crater boolean comes back
as true or false
- include boundary cases for all input
The peak kinetic energy loss per km (airburst)
The burst altitude (airburst)
The total kinetic energy loss at burst (airburst)
The mass and speed of the remnants of the asteroid that strike(s) the ground (airburst and cratering)
'''


import eroscode
import numpy as np

def initial_state_array():
    # [v_init, m_init, theta_init, z_init, x_init, r_init]

    state = np.array([0, 2000, 70, 1e5, 0, 3])

    return state

print(initial_state_array())

eroscode.main()
