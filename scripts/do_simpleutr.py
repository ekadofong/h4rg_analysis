import sys
if 'matplotlib' not in sys.modules:
    import matplotlib
    matplotlib.use('Agg')

import os
import psutil
import datetime
import numpy as np
from ics.hxutils import hxramp
from h4rg_analysis import ramputils,io

bestbaselines = open ('../data/paths/best_ever_baseline_darks_8_21_2022.txt', 'r').read().splitlines()[1:]
_DEFAULT_OUTPUTDIR = '../data/output/'
scratchdir = '/scratch/gpfs/kadofong/h4rg_scratch/erranalysis/'

def badprintdebug ( text ):
    print(text)
    sys.stdout.flush ()

def utrstats ( cube, slicey=None, slicex=None ):
    if slicey is None:
        slicey = slice(None)
    if slicex is None:
        slicex = slice(None)
        
    input_cube = cube[:,slicey,slicex].astype('f4')
    # \\ cut and THEN turn to f4
    out = ramputils.utr_expectedflux ( input_cube )
    flattened = input_cube - out[0] 
    var = np.var(flattened, axis=0)
    return var, np.array(out[1])

def logn ( text, start, tot_start ):
    now = datetime.datetime.now ()
    elapsed = (now - start).seconds / 60.
    tot_elapsed = (now - tot_start).seconds / 60.
    print(f'{text}. {elapsed:.2f} min elapsed ({tot_elapsed:.2f} total)')
    memusage ()
    print('====')
    sys.stdout.flush ()
    
def memusage ():
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss/1e9) 

def singleton ( ramp, frametype, tag=None, r0=0, r1=50, nbins=4, savedir=None, verbose=True ):
    if savedir is None:
        savedir = _DEFAULT_OUTPUTDIR
    if frametype == 'irp':
        frame_label = 'IRP'
        tot_ref = io.irpStack_memmap (ramp, r0=r0, r1=r1)
    elif frametype == 'data':
        frame_label = 'data'
        tot_ref = ramp.dataStack (r0=r0, r1=r1)
    if tag is None:
        tag = f'RND{np.random.randint(0,1000)}'
    
    
    if verbose:
        print (f'starting {tag}')
        sys.stdout.flush()

    step = ramp.ncols // nbins
    sl_indices = np.concatenate ( [np.arange(0, ramp.ncols, step)[:-1], [ramp.ncols+1]] )
    slices = [ slice(sl_indices[idx], sl_indices[idx+1]) for idx in range(sl_indices.size-1) ]
    
    var_arr = np.zeros([ramp.ncols, ramp.nrows] )
    param_arr = np.zeros([2, ramp.ncols, ramp.nrows])
    if not os.path.exists (f"{savedir}{tag}"):
        os.mkdir(f"{savedir}{tag}")
    prefix = f'{tag}_{frame_label}_S{r0}-E{r1}'    
    
    j=0
    total_start = datetime.datetime.now ()
    for cslice_y in slices:
        for cslice_x in slices:
            individ_start = datetime.datetime.now ()
            ref = tot_ref[:,cslice_y, cslice_x] #io.irp_sliced ( ramp, slicey=cslice_y, slicex=cslice_x )
            logn ( f'{frame_label} loaded', individ_start, total_start )
            
            
            #\\ var, params = utrstats ( ref )
            #\\ logn ( 'Stats computed', individ_start, total_start )
            #\\ 
            #\\ var_arr[cslice_y, cslice_x] = var
            #\\ param_arr[:, cslice_y, cslice_x] = params
            #\\ logn(f'''Completed :: {cslice_y}
            #\\  {cslice_x}''', individ_start, total_start )
            #\\ 
            #\\ np.save ( f'{scratchdir}/{prefix}_var{j}.npy', var )
            #\\ np.save ( f'{scratchdir}/{prefix}_utrparams{j}.npy', params )
            #\\ del ref, var, params
            j+=1
    
    np.save ( f'{savedir}{prefix}_var.npy', var_arr )
    np.save( f'{savedir}{prefix}_utrparam.npy', param_arr )

if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
    
    if idx < (len(bestbaselines)//2-1):
        worklist = bestbaselines[idx*2:idx*2+2]
    else:
        worklist = bestbaselines[idx*2:]
    for task in worklist:
        pt = io.jhu2tiger ( task )
        ramp = hxramp.HxRamp( pt )     
        tag = os.path.basename(pt).strip('.fits')   
        if ramp.interleaveOffset > 0:
            singleton ( ramp, 'irp', tag=tag )
        #singleton ( ramp, 'data', tag=tag)
        
    
        