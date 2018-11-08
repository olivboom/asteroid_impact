import initialisation
import eroscode


#set parameters

initialisation.set_parameters("Earth", analytical=True)
initialisation.set_variables("Analytical") #include assert to prevent misspelling

print("done")

eroscode.main()
vvv = eroscode.final_state

t, v,m,theta, z,x, KE,r, burst_index = vvv



