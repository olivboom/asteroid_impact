import eroscode.py

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    return array[np.searchsorted(array, value, side="left")]

def eros_test1():
    # same main method, just change init values
	v_init=19E3
	r_init=19.5/2
	rho_init=3.3E3
	m_init=mass(rho_init,r_init)
	theta_init=deg_to_rad(20)
    state0 = np.array([v_init, m_init, theta_init, z_init, x_init, r_init])
	#IndentationError: unindent does not match any outer indentation level
    dt = 0.1
    t = np.arange(0.0, 40.0, dt)

    # =============================================================================
    #     SOLVE_IVP
    # =============================================================================

    states = solve_ivp(ode_solver, (0, 50), state0, t_eval=t, method='RK45')  # need to fix the f tomorrow
    v = states.y[0]
    m = states.y[1]
    theta = states.y[2]
    z = states.y[3]
    x = states.y[4]
    r = states.y[5]
    KE = 0.5 * m * v ** 2
	# consider the explosion energy of one ton of tnt is equal to 4.184E9 Joule
	Mt=4.18*E15
	# reference value for Chelyabinsk energy is around KE=0.55E6*4.18E9
	# and burst height (reference height) is around z=30m
	# Source: Collins et. al.: Simple airblast models of impact airbursts
	z_ref=30
	t_ref=np.searchsorted(z, zref, side="left")
	KE_ref=0.55*Mt
    KE_est=KE[t_ref]
    assert 0.2*Mt <= KE_est <= 0.8*Mt
'''
Chelyabinsk/Tunguska
KE[Mt]=0.55,9.4 = ... J
d[m]=19.5,50
rho[kg m-3]=3.3E3,3.E3
v[m s]=19E3,20E3
theta[deg]=20,45
Y[MPa]=2,1=...Pa (careful with SI units, Pa=N/mm^2?)

Ablation Parameters KH=1.4*10-8kg/J
Cd=2
H=8.E3
p0=1.22kg/m3

    assert
'''