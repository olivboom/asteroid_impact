import eroscode
import run
import initialisation
import numpy as np

def test_speed1():
    # assert that v is decreasing
    final_state = run.run_asteroid(planet="Earth", 
        asteroid = "Chelyabinsk", show = False, anal_assumption = False, tol = 1e-8)
    v =  final_state[0]
    assert all(np.around(np.diff(v)))<=0
    
    
def test_speed2():
    # assert that v is bounded
    final_state = run.run_asteroid(planet="Earth", 
        asteroid = "Chelyabinsk", show = False, anal_assumption = False, tol = 1e-8)
    v = final_state[0]
    assert all(v >= 0) and all(v<=100E3)
    
    
def test_mass():
    # assert that m is bounded
    final_state = run.run_asteroid(planet="Earth", 
        asteroid = "Chelyabinsk", show = False, anal_assumption = False, tol = 1e-8)
    m = final_state[1]
    assert all(0 < m) and all(m <= 1E9)
    
    
def test_alt1():
    # assert that altitude is bounded
    final_state=run.run_asteroid(planet="Earth", 
        asteroid = "Chelyabinsk", show =False, anal_assumption = False, tol = 1e-8)
    z=final_state[4]
    assert 0<=all(z)<=1E5
    
    
def test_alt2():
    # assert that altitude is decreasing
    final_state = eroscode.main()
    final_state=run.run_asteroid(planet="Earth", 
        asteroid = "Chelyabinsk", show =False, anal_assumption = False, tol = 1e-8)
    z=final_state[4]
    assert all(np.diff(z) <= 0)
    
    
def test_alt3():
    # assert crater event for large asteroids
    final_state = eroscode.main()
    final_state=run.run_asteroid(planet="Earth", 
        asteroid = "Chelyabinsk", show =False, anal_assumption = False, tol = 1e-8)
    r=final_state[7]
    burst_index = final_state[8]
    if r[0]>100:
        assert burst_index == 0
    
    
def test_dist2():
    # assert x is increasing
    final_state=run.run_asteroid(planet="Earth", 
        asteroid = "Chelyabinsk", show =False, anal_assumption = False, tol = 1e-8)
    x=final_state[5]
    assert all(np.diff(x) >= 0)
    
    
def test_KE():
    # assert KE is bounded
    final_state=run.run_asteroid(planet="Earth", 
        asteroid = "Chelyabinsk", show =False, anal_assumption = False, tol = 1e-8)
    KE=final_state[6]
    assert np.all(KE/(4.184E12)<=10**4)
    
    
def test_Chelyabinsk():
    final_state=run.run_asteroid(planet="Earth", 
        asteroid = "Chelyabinsk", show =False, anal_assumption = False, tol = 1e-8)
    KE=final_state[6]
    z = final_state[4]
    assert np.all(abs(np.diff(KE)/np.diff(z))*1e3 / 4.184E12<=10**4)
    
    
def test_Tunguska():
    #Call Tungaska Parameters
    final_state=run.run_asteroid(planet="Earth", 
        asteroid = "Chelyabinsk", show =False, anal_assumption = False, tol = 1e-8)
    KE=final_state[6]
    z = final_state[4]
    assert all(abs(np.diff(KE)/np.diff(z))*1e3 / 4.184E12<=10**4)