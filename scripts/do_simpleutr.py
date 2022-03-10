import sys
if 'matplotlib' not in sys.modules:
    import matplotlib
    matplotlib.use('Agg')

import os
import numpy as np
from ics.hxutils import hxramp
from h4rg_analysis import ramputils,io

bestbaselines = open ('../data/paths/best_ever_baseline_darks_8_21_2022.txt', 'r').read().splitlines()[1:]
_DEFAULT_OUTPUTDIR = '../data/output/'

def singleton ( path, start=None, stop=None, savedir=None):
    if savedir is None:
        savedir = _DEFAULT_OUTPUTDIR
        
    pt = io.jhu2tiger ( path )
    ramp = hxramp.HxRamp( pt )
    ref = ramp.irpStack (dtype='f4')
    data = ramp.dataStack(dtype='f4')
    
    
    ref_pkg = ramputils.utr_expectedflux ( ref, start=start, stop=stop ) # ref_eflux, (ref_m, ref_b)
    data_pkg = ramputils.utr_expectedflux ( data, star=start, stop=stop ) # data_eflux, (data_m, data_b)
    
    #\\ save reference and parameter frames
    obs_l = [ref,data]
    out_l = [ref_pkg, data_pkg]
    lbl_l = ['IRP','data']
    tag = os.path.basename(pt).strip('.fits')

    for idx in range(2):
        obs = obs_l[idx]
        out = out_l[idx]
        lbl = lbl_l[idx]
        flattened = obs[:,start:stop,start:stop] - out[0]
        var = np.var(flattened, axis=0)
        np.save(f'{savedir}/{tag}_{lbl}_var.npy', var)        
        np.save(f'{savedir}/{tag}_{lbl}_utrparam.npy', np.array(out[1]) )
        
        
        