import eroscode.py

def test_speed(v):
    # assert that v is decreasing
    assert all(np.around(np.diff(v))<=0)
    
    
def test_mass(m):
    # assert that m is decreasing
    assert all(np.around(np.diff(m)) < 0)
    
    
def test_alt(z):
    # assert that altitude is decreasing
    assert all(np.diff(z) <= 0)
    
    
def test_alt2(r,):
    # assert crater event for large astroids
    if r[0]>100:
        assert burst_index==0
    else:
        assert burst_index==1
     
def test_dist(x):
    # assert x is increasing
    assert all(np.diff(x) >= 0)
    
    
def test_KE(KE,z):
    # assert KE is bounded
    assert all(0<=KE/z*1e3 / 4.18E12) and all(KE/z*1e3 / 4.18E12<=10**4)
