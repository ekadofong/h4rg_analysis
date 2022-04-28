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
    
    se_b = s * np.sqrt ( 1./cube.shape[0] * xs.mean()**2 / ss_xx )
    se_a = s / np.sqrt(ss_xx)
    
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

def correct_IRPpcv (irp):
    '''
    Correct the per-channel variation (vertical stridation) in the IRP frames.
    There are 32 channels across the frame (128 pix each)
    '''
    irpf = irp.astype(float)
    xs = np.arange(irpf.shape[1])
    
    bins = np.arange(0,xs.max() + 1, 128)
    midpts = 0.5*(bins[1:]+bins[:-1])
    dt = np.digitize (xs, bins)
    
    # \\ take an initial per-channel median
    rowmed = np.median ( irpf, axis=0 )
    colmed = ndimage.median ( rowmed, labels=dt, index=np.unique(dt) )
    recrow = recast ( colmed, dt )
    
    # \\ sigma clip @ 4 sigma and remake median
    rowdiff = rowmed - recrow
    sclip = stats.sigmaclip(rowdiff)
    cpd_diff = irpf.copy ()
    cpd_diff[:,(rowdiff>sclip.upper)|(rowdiff<sclip.lower)] = np.NaN
    cpd_rowmed = np.nanmedian(cpd_diff, axis=0)
    cpd_colmed = ndimage.median ( cpd_rowmed, labels=dt, index=np.unique(dt) )    
    cpd_recast = recast ( cpd_colmed, dt)
    normalized = np.ones(irpf.shape, dtype=float)*cpd_recast
    
    return irpf - normalized, cpd_colmed

def colsub ( arr, colwidth=128):
    arrf = arr.astype(float)
    xs = np.arange(arrf.shape[1])
    
    bins = np.arange(0,xs.max() + 1, colwidth)
    midpts = 0.5*(bins[1:]+bins[:-1])
    dt = np.digitize (xs, bins)
    
    # \\ take an initial per-channel median
    rowmed = np.median ( arrf, axis=0 )
    colmed = ndimage.median ( rowmed, labels=dt, index=np.unique(dt) )
    recrow = recast ( colmed, dt )
    
    # \\ sigma clip @ 4 sigma and remake median
    rowdiff = rowmed - recrow
    sclip = stats.sigmaclip(rowdiff)
    cpd_diff = arrf.copy ()
    cpd_diff[:,(rowdiff>sclip.upper)|(rowdiff<sclip.lower)] = np.NaN
    cpd_rowmed = np.nanmedian(cpd_diff, axis=0)
    cpd_colmed = ndimage.median ( cpd_rowmed, labels=dt, index=np.unique(dt) )    
    cpd_recast = recast ( cpd_colmed, dt)
    normalized = np.ones(arrf.shape, dtype=float)*cpd_recast
    
    return arrf - normalized, cpd_colmed    

def roughclip ( img, sigma=3., fill=np.NaN ):
    sigclip = stats.sigmaclip ( img[2], low=sigma, high=sigma)
    lower = sigclip.lower
    upper = sigclip.upper
    mask = (img>upper)|(img<lower)
    return np.where(mask, fill, img)

def flat_statistics ( cube, gain=1., readtime=1., nbins=20, maxrate=1. ):
    diff = np.diff(cube.astype(float), axis=0)
    clipped_diff = roughclip (diff)
    Nreads=np.logspace(1, np.log10(cube.shape[0]), nbins).astype(int)
    statarr = np.zeros([2, nbins, *clipped_diff.shape[1:]])
    for idx,N in enumerate(Nreads):
        mrate = np.nanmean(clipped_diff[:N], axis=0) # ADU/read
        mrate *= gain / readtime # ADU / read * e / ADU * read / s
        srate = np.nanstd(clipped_diff[:N], axis=0) # ADU/read
        srate *= gain / readtime
        statarr[0,idx] = mrate
        statarr[1,idx] = srate  
        
    if maxrate is not None:
        clipper = abs(statarr[0,-1]) > maxrate
        statarr[0,:,clipper] = np.NaN
        statarr[1,:,clipper] = np.NaN
    return statarr  

def ps2D ( img ):
    from turbustat.statistics import pspec
    
    ps2D = np.power(np.fft.fftshift ( pspec.rfft_to_fft(img) ),2.)    
    yfreqs = np.fft.fftshift(np.fft.fftfreq(img.shape[0]))
    xfreqs = np.fft.fftshift(np.fft.fftfreq(img.shape[1]))
    return ps2D, xfreqs, yfreqs


def compute_uncertainties (cubeA, cubeB, xs=None, gain=None, exptime=None, expected=None):
    '''
    Compute statistics for difference in OLS rate for 2 ramps
    rn_arr[0] = RMS difference between rampA and rampB
    rn_arr[1] = RMS difference between rampA and expected rate (only meaningful for fake data)
    rn_arr[2] = OLS SE on slope estimate of rampA
    rn_arr[3] = mean rate of rampA
    rn_arr[3] = median rate of rampA
    (All quantities are sigmaclipped)
    '''
    if xs is None:
        xs = np.logspace(np.log10(5.),np.log10(cubeA.shape[0]-1),20).astype(int)
    if gain is None:
        print('Caution! Gain=1')    
        gain = 1.
    if exptime is None:
        print('Caution! Exptime=1 sec')
        exptime = 1.
    if expected is None:
        expected = np.NaN
    unit_conversion = gain / exptime # e/ADU / s 
    
    rn_arr = np.zeros([5,xs.size])
    for idx, read_idx in enumerate(xs):
        ca = cubeA[:(read_idx+1)].astype(float)
        cb = cubeB[:(read_idx+1)].astype(float)
        _,rhatA,rhatA_se = utr_expectedflux ( ca )
        _,rhatB,rhatB_se = utr_expectedflux ( cb )
        _rdiff = (rhatA[0] - rhatB[0]) * unit_conversion # ADU/read * e/ADU * read/s
        clipper = stats.sigmaclip(_rdiff, low=3, high=3)    
        rdiff = np.where((_rdiff<clipper.upper)&(_rdiff>clipper.lower), _rdiff, np.NaN)
        detarea = 2.*np.isfinite(rdiff).sum()
        rate_noise = np.sqrt( np.nansum(rdiff**2) / detarea )
        rn_arr[0,idx] = rate_noise
        
        ediff = (rhatA[0] - expected) * unit_conversion # ADU/read * e/ADU * read/s
        eclipper = stats.sigmaclip(ediff, low=3, high=3)
        ediff = np.where((ediff<eclipper.upper)&(ediff>eclipper.lower), ediff, np.NaN)
        rn_arr[1,idx] = np.sqrt ( np.nansum(ediff**2) / (2.*np.isfinite(ediff).sum()) )
        

        se = np.where((_rdiff<clipper.upper)&(_rdiff>clipper.lower), rhatB_se, np.NaN)
        rn_arr[2,idx] = np.nanmedian(se[0])   
        rclipper = stats.sigmaclip(rhatA[0], low=3,high=3)
        mask = (_rdiff<clipper.upper)&(_rdiff>clipper.lower) #(rhatA[0] < rclipper.upper )&( rhatA[0] > rclipper.lower )
        rn_arr[3,idx] = np.nanmean(np.where(mask, rhatA[0], np.NaN)) * unit_conversion
        rn_arr[4,idx] = np.nanmedian(np.where(mask, rhatA[0], np.NaN)) * unit_conversion
        
    return xs, rn_arr