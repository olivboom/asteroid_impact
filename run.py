import initialisation
import eroscode
import plotting_analytical
import matplotlib.pyplot as plt
import numpy as np
import plots
import statistical_ensemble as se



def settolerance(tol):
    eroscode.tol = tol




def run_asteroid(planet="Earth", asteroid = "Tunguska", show =True):
    initialisation.set_parameters(planet, analytical_assumption=False)
    if asteroid == "Analytical Assumptions":
        initialisation.set_parameters(planet, analytical_assumption=True)
    settolerance(1e-8)
    state = initialisation.set_variables(asteroid)
    eroscode.initial_state = state
    eroscode.ensemble = False
    eroscode.main()
    final_state = eroscode.main()
    
    
    if show == True:
        plots.plot_5_graphs(final_state)
    
    return final_state

def run_custom(variables, planet="Earth",show =True):
    """
    input:
    variables = array type
                v_init, m_init, theta_init, z_init, x_init, r_init
                
    output:
        
    """
    
    initialisation.set_parameters(planet, analytical_assumption=False)
    

    settolerance(1e-8)
    state = variables
    eroscode.initial_state = state
    eroscode.ensemble = False
    eroscode.main()
    final_state = eroscode.main()
    
    
    if show == True:
        plots.plot_5_graphs(final_state)
    
    return final_state

def run_ensemble(planet="Earth", asteroid = "Tunguska" ,analytical=False, ensemble=False, num = 10, show =True):
    final_states = []
    KE = []
    height = []
    initialisation.set_parameters(planet, analytical_assumption=False)
    settolerance(1e-8)
    eroscode.ensemble = True
    
    states = se.confidence_prediction(num)
    for n,i in enumerate(range(states.shape[1])):
        if n%50 ==0:
            print(n)
        state = states[:,i]
        eroscode.initial_state = state
        eroscode.main()
        final_state = eroscode.main()
        ke_max_value, ke_max_height = eroscode.find_ke_max(final_state)

        KE.append(ke_max_value)
        height.append(ke_max_height)
        final_states.append(final_state)
        
    if show == True:
        plots.ensemble_scatter(KE,height)
        
    return final_states

        
    

#def analytical_compare(asteroid = "Tunguska"):
#    variables = initialisation.set_variables(asteroid)
#    parameters_ = initialisation.set_parameters_analytical_solution()
#    plotting_analytical.analytical(variables)
    
#    plotting_analytical.analytical(variables)
    
#initialisation.set_parameters_analytical_solution()
#print(plotting_analytical.returnH())
#analytical_compare()
#run_ensemble(num = 1000)
#run_custom(variables = np.array([20e3, 11e6, deg_to_rad(45), 100e3, 0, 25]))

#
#    
