import numpy as np


def utr_rdt (yvec, axis=0):
    '''
    Estimate of Delta_t * R for up-the-ramp
    '''
    xs = np.arange(yvec.shape[0])
    n = yvec.shape[0]
    t1a = n*np.sum(xs[:,np.newaxis,np.newaxis]*yvec, axis=axis) 
    t1b = np.sum(xs)*np.sum(yvec, axis=axis)
    t1 = t1a - t1b
    t2 = n*np.sum(xs**2, axis=axis) - np.sum(xs, axis=axis)**2
    return t1/t2


def utr_expectedflux ( cube, start=None, stop=None, bmethod='naive' ):    
    if bmethod == 'naive':
        b = cube[:10].mean(axis=0)
        
    rdt_estimate = utr_rdt ( cube[:,start:stop,start:stop] )
    xs = np.arange(cube.shape[0])
    expected_flux = (rdt_estimate[np.newaxis,start:stop,start:stop] * xs[:,np.newaxis,np.newaxis]) 
    expected_flux += b[np.newaxis,start:stop,start:stop]
    return expected_flux