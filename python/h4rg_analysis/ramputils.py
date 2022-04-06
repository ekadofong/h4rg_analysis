import numpy as np
from scipy import stats
from scipy import ndimage

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


def utr_expectedflux ( cube, bmethod='naive', return_param=True ):    
    rdt_estimate = utr_rdt ( cube )
    xs = np.arange(cube.shape[0])
    expected_flux = (rdt_estimate[np.newaxis] * xs[:,np.newaxis,np.newaxis]) 
    cmean = cube.mean(axis=0)
    if bmethod=='naive':
        b = cmean - expected_flux.mean(axis=0)
        
    ss_xx = np.sum((xs - np.mean(xs))**2)
    ss_yy = np.sum((cube - cmean)**2, axis=0)
    ss_xy = np.sum((xs - np.mean(xs))[:,np.newaxis,np.newaxis]*(cube-cmean), axis=0)
    
    s = np.sqrt ( (ss_yy - ss_xy**2 / ss_xx)/(cube.shape[0]- 2.) )
    
    se_a = s * np.sqrt ( 1./cube.shape[0] * xs.mean()**2 / ss_xx )
    se_b = s / np.sqrt(ss_xx)
    
    expected_flux += b[np.newaxis]
    if return_param:
        return expected_flux, (rdt_estimate, b), (se_a, se_b)
    else:
        return expected_flux
    
def recast (colmed, dt):
    recast = np.ones(dt.size, dtype=float)
    for idx,dindex in enumerate(np.unique(dt)):
        recast[dt==dindex] = colmed[idx]    
    return recast

def correct_IRPpcv (irp1, irp0):
    '''
    Correct the per-channel variation (vertical stridation) in the IRP frames.
    There are 32 channels across the frame (128 pix each)
    '''
    irpdiff = irp1.astype(float)-irp0.astype(float)
    xs = np.arange(irpdiff.shape[1])
    
    bins = np.arange(0,xs.max() + 2, 128)
    midpts = 0.5*(bins[1:]+bins[:-1])
    dt = np.digitize (xs, bins)
    
    # \\ take an initial per-channel median
    rowmed = np.median ( irpdiff, axis=0 )
    colmed = ndimage.median ( rowmed, labels=dt, index=np.unique(dt) )
    recrow = recast ( colmed, dt )
    
    # \\ sigma clip @ 4 sigma and remake median
    rowdiff = rowmed - recrow
    sclip = stats.sigmaclip(rowdiff)
    cpd_diff = irpdiff.copy ()
    cpd_diff[:,(rowdiff>sclip.upper)|(rowdiff<sclip.lower)] = np.NaN
    cpd_rowmed = np.nanmedian(cpd_diff, axis=0)
    cpd_colmed = ndimage.median ( cpd_rowmed, labels=dt, index=np.unique(dt) )    
    cpd_recast = recast ( cpd_colmed, dt)
    normalized = np.ones(irpdiff.shape, dtype=float)*cpd_recast
    
    return irpdiff - normalized