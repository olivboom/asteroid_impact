import eroscode.py

def test_speed():
    # assert that v is decreasing
    v,m,theta,z,x,KE,r = main()
    assert all(np.around(np.diff(v))<=0)
    
    
def test_mass1():
    # assert that mass is not exceeded
    v,m,theta,z,x,KE,r = main()
    assert all(m <= m[0])
    
    
def test_mass2():
    # assert that m is decreasing
    v,m,theta,z,x,KE,r = main()
    assert all(np.around(np.diff(m)) < 0)
    
    
def test_alt1():
    # assert that z is not exceeded
    v,m,theta,z,x,KE,r = main()
    assert all(z <= z[0])
    
    
def test_alt2():
    # assert that altitude is decreasing
    v,m,theta,z,x,KE,r = main()
    assert all(np.diff(z) <= 0)
    
    
def test_alt3():
    # assert crater event for large astroids
    v,m,theta,z,x,KE,r = main()
    if r[0]>100:
        assert z[-1] <= 0
    else:
        assert all(z > 0)
    
def test_dist1():
    # assert x is not underceeded
    v,m,theta,z,x,KE,r = main()
    assert all(x >= 0)
    
    
def test_dist2():
    # assert x is increasing
    v,m,theta,z,x,KE,r=main()
    assert all(np.diff(x) >= 0)
    
    
def test_KE():
    # assert KE is bounded
    v,m,theta,z,x,KE,r=main()
    assert all(0<=KE/z*1e3 / 4.18E12) and all(KE/z*1e3 / 4.18E12<=10**4)