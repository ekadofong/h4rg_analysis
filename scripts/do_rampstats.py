import sys
if 'matplotlib' not in sys.modules:
    import matplotlib
    matplotlib.use('Agg')

import os
import psutil
import datetime
import tempfile
import numpy as np
from astropy.io import fits
#import matplotlib.pyplot as plt
from ics.hxutils import hxramp
from h4rg_analysis import io, ramputils

bestbaselines = open ('../data/paths/best_ever_baseline_darks_8_21_2022.txt', 'r').read().splitlines()[1:]
_DEFAULT_OUTPUTDIR = '../data/output/'
_DEFAULT_TEMPDIR = './' #'/scratch/gpfs/kadofong/h4rg_scratch/tmp/' # XXX trying to figure out slow read times
gain = 3.2 * 5./7. # e-/ADU, measured by JHU for detector 18660

if os.path.exists('/scratch/gpfs/kadofong/'):
    scratchdir = '/scratch/gpfs/kadofong/h4rg_scratch/erranalysis/'
else:
    scratchdir = './'


def memusage ():
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss/1e9) 

def logn ( text, start, tot_start ):
    now = datetime.datetime.now ()
    elapsed = (now - start).seconds / 60.
    tot_elapsed = (now - tot_start).seconds / 60.
    print(f'{text}. {elapsed:.2f} min elapsed ({tot_elapsed:.2f} total)')
    memusage ()
    print('====')
    sys.stdout.flush ()
    


def singleton (fitsfile, 
               frame_label, 
               gain,               
               tag=None, 
               r0=0, 
               r1=-1, 
               nbins=10,
               savedir=None, 
               verbose=True, 
               total_start=None,
               clobber=False,):
    if total_start is None:
        total_start = datetime.datetime.now ()
    individ_start = datetime.datetime.now ()
    if savedir is None:
        savedir = _DEFAULT_OUTPUTDIR        
    if tag is None:
        tag = f'RND{np.random.randint(0,1000)}'

    # \\ check to see if this has already been run
    prefix = f'{tag}_{frame_label}_S{r0}-E{r1}'
    if (not clobber) and (os.path.exists(f'{savedir}{prefix}_var.npy')):        
        print(f'{prefix} already run.')
        sys.stdout.flush ()
        return 0
    
    if verbose:
        logn (f'starting {tag} [{frame_label}]', individ_start, total_start)

    rampfits = fits.open(fitsfile)
    nrows = rampfits[1].header['NAXIS1']
    ncols = rampfits[1].header['NAXIS2']
    bins = np.logspace(1, np.log10(300), 20).astype(int)
    
    step = ncols // nbins
    sl_indices = np.concatenate ( [np.arange(0, ncols, step), [ncols+1]] )
    slices = [ slice(sl_indices[idx], sl_indices[idx+1]) for idx in range(sl_indices.size-1) ]
    
    var_arr = np.zeros([bins.size, 2, ncols, nrows] )
    param_arr = np.zeros([bins.size, 2, ncols, nrows])
    exptime = rampfits[0].header['W_FRMTIM'] # sec
    unit_conversion = gain / exptime # e/ADU / s 

    
    j=0    
    for cslice_y in slices:
        for cslice_x in slices:
            #if not clobber and os.path.exists(f'{scratchdir}/{prefix}_utrse{j}.npy'):
            #    var_arr[:,cslice_y,cslice_x] = np.load(f'{scratchdir}/{prefix}_utrse{j}.npy')
            #    param_arr[:,cslice_y,cslice_x] = np.load(f'{scratchdir}/{prefix}_utrparams{j}.npy')
            #    continue
            individ_start = datetime.datetime.now ()

            cube = io.readcube ( rampfits, frame_label, slicey=cslice_y, slicex=cslice_x, timeit=True )
            sys.stdout.flush ()
            
            logn(f'''Loaded :: {cslice_y}, {cslice_x}''', individ_start, total_start )
            
            for idx,endbin in enumerate(bins):
                _,rhat,rhat_se = ramputils.utr_expectedflux ( cube[:endbin] )
                                
                var_arr[idx, 0, cslice_y, cslice_x] = rhat_se[0] * unit_conversion
                var_arr[idx, 1, cslice_y, cslice_x] = rhat_se[1] * gain
                param_arr[idx, 0, cslice_y, cslice_x] = rhat[0] * unit_conversion        
                param_arr[idx, 1, cslice_y, cslice_x] = rhat[1] * gain
                if verbose:
                    logn(f'''Completed :: {cslice_y}, {cslice_x}''', individ_start, total_start )
            
            np.save ( f'{scratchdir}/{prefix}_utrse{j}.npy', var_arr[:, cslice_y, cslice_x] )
            np.save ( f'{scratchdir}/{prefix}_utrparams{j}.npy', param_arr[:, cslice_y, cslice_x] )
            
            del rhat, rhat_se
            j+=1
    
    np.save ( f'{savedir}{prefix}_var.npy', var_arr )
    np.save( f'{savedir}{prefix}_utrparam.npy', param_arr )
    logn (f'Completed {prefix}',  individ_start, total_start)

def singleton_memmap ( ramp, frame_label, gain,               
               tag=None, 
               r0=0, 
               r1=-1, 
               nbins=4, 
               savedir=None, 
               verbose=True, 
               total_start=None,
               clobber=False,
               tempdir=None ):
    if total_start is None:
        total_start = datetime.datetime.now ()
    individ_start = datetime.datetime.now ()
    if savedir is None:
        savedir = _DEFAULT_OUTPUTDIR        
    if tag is None:
        tag = f'RND{np.random.randint(0,1000)}'
    if tempdir is None:
        tempdir = _DEFAULT_TEMPDIR
        
    # \\ set function to get frames
    if frame_label == 'data':
        readfn = io.dataStack_memmap
    elif frame_label == 'corr':
        readfn = io.corrStack_memmap
    elif frame_label == 'irp':
        readfn = io.irpStack_memmap
        
    # \\ check to see if this has already been run
    prefix = f'{tag}_{frame_label}_S{r0}-E{r1}'
    if (not clobber) and (os.path.exists(f'{savedir}{prefix}_var.npy')):        
        print(f'{prefix} already run.')
        sys.stdout.flush ()
        return 0
    
    if verbose:
        logn (f'starting {tag} [{frame_label}]', individ_start, total_start)

    step = ramp.ncols // nbins
    sl_indices = np.concatenate ( [np.arange(0, ramp.ncols, step), [ramp.ncols+1]] )
    slices = [ slice(sl_indices[idx], sl_indices[idx+1]) for idx in range(sl_indices.size-1) ]
    bins = np.logspace(1, np.log10(300), 20).astype(int)
    
    var_arr = np.zeros([bins.size, 2, ramp.ncols, ramp.nrows] )
    param_arr = np.zeros([bins.size, 2, ramp.ncols, ramp.nrows])
    exptime = ramp.header()['W_FRMTIM'] # sec
    unit_conversion = gain / exptime # e/ADU / s 
    
    
    j=0    
    for cslice_y in slices:
        for cslice_x in slices:
            if not clobber and os.path.exists(f'{scratchdir}/{prefix}_utrse{j}.npy'):
                var_arr[:,cslice_y,cslice_x] = np.load(f'{scratchdir}/{prefix}_utrse{j}.npy')
                param_arr[:,cslice_y,cslice_x] = np.load(f'{scratchdir}/{prefix}_utrparams{j}.npy')
                continue
            individ_start = datetime.datetime.now ()
            tfile=tempfile.NamedTemporaryFile(dir=tempdir)
            cube, _ = readfn ( ramp, filename=tfile, slicey=cslice_y, slicex=cslice_x )
            logn(f'''Loaded :: {cslice_y}, {cslice_x}''', individ_start, total_start )
            for idx,endbin in enumerate(bins):
                _,rhat,rhat_se = ramputils.utr_expectedflux ( cube[:endbin] )
                                
                var_arr[idx, 0, cslice_y, cslice_x] = rhat_se[0] * unit_conversion
                var_arr[idx, 1, cslice_y, cslice_x] = rhat_se[1] * gain
                param_arr[idx, 0, cslice_y, cslice_x] = rhat[0] * unit_conversion        
                param_arr[idx, 1, cslice_y, cslice_x] = rhat[1] * gain
                if verbose:
                    logn(f'''Completed :: {cslice_y}, {cslice_x}''', individ_start, total_start )
            
            np.save ( f'{scratchdir}/{prefix}_utrse{j}.npy', var_arr[:, cslice_y, cslice_x] )
            np.save ( f'{scratchdir}/{prefix}_utrparams{j}.npy', param_arr[:, cslice_y, cslice_x] )
            
            del rhat, rhat_se
            j+=1
    
    np.save ( f'{savedir}{prefix}_var.npy', var_arr )
    np.save( f'{savedir}{prefix}_utrparam.npy', param_arr )
    logn (f'Completed {prefix}',  individ_start, total_start)
    
    
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
    chunk = 1
    if idx < (len(bestbaselines)//chunk-1):
        worklist = bestbaselines[idx*chunk:idx*chunk+chunk]
    else:
        worklist = bestbaselines[idx*chunk:]
        
    total_start = datetime.datetime.now ()
    for task in worklist:
        pt = io.jhu2tiger ( task )
        ramp = hxramp.HxRamp( pt )     
        tag = os.path.basename(pt).strip('.fits')           
        kwargs = {'tag':tag,'total_start':total_start}
        singleton ( ramp, 'data', gain=gain, **kwargs)
        singleton ( ramp, 'corr', gain=gain, **kwargs)
        if ramp.interleaveOffset > 0:
            singleton ( ramp, 'irp', gain=gain, **kwargs)
elif __name__=='__main__':
    use_memmap=True
    total_start = datetime.datetime.now ()
    for task in bestbaselines[17:18]: # XXX read speed
        pt = io.jhu2tiger ( task )
        #ramp = hxramp.HxRamp( pt )     
        
        tag = os.path.basename(pt).strip('.fits') 
        if not use_memmap:          
            kwargs = {'tag':tag,'total_start':total_start}
            singleton ( pt, 'data', gain=gain, **kwargs)
            singleton ( pt, 'corr', gain=gain, **kwargs)
            singleton ( pt, 'irp', gain=gain, **kwargs)
        else:    
            pt = io.jhu2tiger ( task )
            ramp = hxramp.HxRamp( pt )     
            tag = os.path.basename(pt).strip('.fits')           
            kwargs = {'tag':tag,'total_start':total_start}
            singleton_memmap ( ramp, 'data', gain=gain, **kwargs)
            singleton_memmap ( ramp, 'corr', gain=gain, **kwargs)
            if ramp.interleaveOffset > 0:
                singleton ( ramp, 'irp', gain=gain, **kwargs)