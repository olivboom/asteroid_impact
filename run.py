import initialisation
import eroscode
import plotting_analytical
import matplotlib.pyplot as plt
import numpy as np






#set parameters
def run(planet="Earth", asteroid = "Ensemble" ,analytical=False, ensemble=False):
    final_states = []
    KE = []
    height = []
    initialisation.set_parameters(planet, analytical_assumption=False)
    if asteroid == "Ensemble":
#        print("asdasd")
        eroscode.ensemble = True
#        print(asteroid)
        states = initialisation.set_variables(asteroid)
        print(states)
        for n,i in enumerate(range(states.shape[1])):
            if n%50 ==0:
                print(n)
            state = states[:,i]
#            print(state)
            eroscode.initial_state = state
            eroscode.main()
            data = eroscode.final_state
            ke_max_value, ke_max_height = eroscode.find_ke_max(data)
            KE.append(ke_max_value)
            height.append(ke_max_height)
            final_states.append(data)
#            print(data)
    else:
        print("dasf")
        state = initialisation.set_variables(asteroid)
        eroscode.initial_state = state
        eroscode.ensemble = False
        eroscode.main()
        data = eroscode.final_state

    return final_states, KE, height

final_states, KE, height = run()

tuples = zip(KE,height)
a = list(tuples)
x,y = np.array(a)[:,0], np.array(a)[:,1]

plt.plot(x,y,'.')
plt.xlabel('KE')


#t, v,m,theta, z,x, ke,r, burst_index, airburst_event = run()
#
#eroscode.plot(t, v, m, z, ke, r, burst_index)


#unitke,z = find_ke_max(data)


