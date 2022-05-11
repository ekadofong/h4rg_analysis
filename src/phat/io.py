import os
import sys
import tempfile
import datetime
import numpy as np
from ics.hxutils import hxramp

def jhu2tiger ( pathtofile ):
    jhu_root = "/data/"
    tiger_root = "/projects/HSC/PFS/JHU/"
    return tiger_root + pathtofile.strip(jhu_root).replace('83.fits', '23.fits')

def read(ramp, frametype, r0=0, r1=-1, slicey=None, slicex=None, timeit=False):
    if slicey is None:
        shape_y = ramp.ncols
    else:
        shape_y = min(ramp.nrows, slicey.stop) - slicey.start
    if slicex is None:
        shape_x = ramp.nrows
    else:
        shape_x = min(ramp.ncols, slicex.stop) - slicex.start
        
    if frametype=='corr':
        fn = ramp.readN
    elif frametype=='data':
        fn = ramp.dataN
    elif frametype=='irp':
        fn = ramp.irpN
            
    r0 = ramp._readIdxToAbsoluteIdx(r0)
    r1 = ramp._readIdxToAbsoluteIdx(r1)
    nreads = r1 - r0 + 1

    stack = np.empty(shape=(nreads,shape_y,shape_x), dtype='f4')
    if timeit:
        start = datetime.datetime.now ()
        reset = datetime.datetime.now ()
    for r_i in range(r0, r1+1):
        read1 = fn(r_i)
        stack[r_i,:,:] = read1[slicey, slicex]
        
        if timeit and r_i%75 == 0:
            lap = datetime.datetime.now () - reset
            elapsed = datetime.datetime.now () - start
            reset = datetime.datetime.now ()
            print ( '%.2f s [total %.2f s]' % (lap.seconds, elapsed.seconds))

    return stack

def irp_sliced ( ramp, r0=0, r1=-1, slicey=None, slicex=None ):
    if slicey is None:
        shape_y = ramp.ncols
    else:
        shape_y = min(ramp.nrows, slicey.stop) - slicey.start
    if slicex is None:
        shape_x = ramp.nrows
    else:
        shape_x = min(ramp.ncols, slicex.stop) - slicex.start
        
    r0 = ramp._readIdxToAbsoluteIdx ( r0 )
    r1 = ramp._readIdxToAbsoluteIdx ( r1 )
    cimage = np.zeros ( [ramp.nreads, shape_y, shape_x], dtype='u2')
    
    for r_i in range(r0, r1+1):
        n = ramp._readIdxToFITSIdx ( r_i )
        extname = f'REF_{n}'
        irpImage = ramp.fits[extname].read ()
        irpImage = hxramp.constructFullIrp(irpImage, ramp.nchan, refPix=ramp.interleaveOffset)
        cimage[r_i] = irpImage[slicey,slicex]    
    return cimage

def irpStack_memmap ( ramp, r0=0, r1=-1, filename=None, slicey=None, slicex=None ):
    if filename is None:        
        filename = tempfile.NamedTemporaryFile()
    if slicey is None:
        shape_y = ramp.nrows
    else:
        shape_y = min(ramp.nrows, slicey.stop) - slicey.start
    if slicex is None:
        shape_x = ramp.ncols
    else:
        shape_x = min(ramp.ncols, slicex.stop) - slicex.start               
        
    r0 = ramp._readIdxToAbsoluteIdx ( r0 )
    r1 = ramp._readIdxToAbsoluteIdx ( r1 )    
    fp = np.memmap ( filename, dtype='u2', mode='w+', shape=(r1-r0+1, shape_y, shape_x )) 
    for r_i in np.arange(r0, r1+1):
        n = ramp._readIdxToFITSIdx ( r_i )
        extname = f'REF_{n}'
        irpImage = ramp.fits[extname].read ()
        irpImage = hxramp.constructFullIrp(irpImage, ramp.nchan, refPix=ramp.interleaveOffset)
        fp[r_i-r0] = irpImage[slicey,slicex]
    fp.flush ()
    del fp
    fread = np.memmap ( filename, dtype='f4', mode='r+', shape=(r1-r0+1, shape_y, shape_x ))
    return fread, filename

def dataStack_memmap ( ramp, r0=0, r1=-1, filename=None, slicey=None, slicex=None ):    
    if filename is None:
        filename = tempfile.NamedTemporaryFile()
    if slicey is None:
        shape_y = ramp.nrows
    else:
        shape_y = min(ramp.nrows, slicey.stop) - slicey.start
    if slicex is None:
        shape_x = ramp.ncols
    else:
        shape_x = min(ramp.ncols, slicex.stop) - slicex.start          
        
    r0 = ramp._readIdxToAbsoluteIdx ( r0 )
    r1 = ramp._readIdxToAbsoluteIdx ( r1 )
    fp = np.memmap ( filename, dtype='u2', mode='w+', shape=(r1-r0+1, shape_y, shape_x )) 
    for r_i in np.arange(r0, r1+1):
        n = ramp._readIdxToFITSIdx ( r_i )
        extname = f'IMAGE_{n}'
        dataImage = ramp.fits[extname].read ()
        fp[r_i-r0] = dataImage[slicey,slicex]
    fp.flush ()
    del fp
    fread = np.memmap ( filename, dtype='f4', mode='r+', shape=(r1-r0+1, shape_y, shape_x ))
    return fread, filename


def corrStack_memmap ( ramp, r0=0, r1=-1, slicey=None, slicex=None, filename=None, read=True):
    if filename is None:
        filename = tempfile.NamedTemporaryFile()
    if slicey is None:
        shape_y = ramp.nrows
    else:
        shape_y = min(ramp.nrows, slicey.stop) - slicey.start
    if slicex is None:
        shape_x = ramp.ncols
    else:
        shape_x = min(ramp.ncols, slicex.stop) - slicex.start        
        
    r0 = ramp._readIdxToAbsoluteIdx ( r0 )
    r1 = ramp._readIdxToAbsoluteIdx ( r1 )
    fp = np.memmap ( filename, dtype='f4', mode='w+', shape=(r1-r0+1, shape_y, shape_x ))        
    
    for r_i in np.arange(r0, r1+1):
        corrImage = ramp.readN(r_i)[slicey,slicex] 
        fp[r_i - r0] = corrImage
        fp.flush ()
    del fp
    if read:
        fread = np.memmap ( filename, dtype='f4', mode='r+', shape=(r1-r0+1, shape_y, shape_x ))
        return fread, filename
    
def readcube ( rampfits, datatype, nreads=300, slicey=None, slicex=None, filename=None, timeit=False):
    nrows = rampfits[1].header['NAXIS1']
    ncols = rampfits[1].header['NAXIS2']
    spacer = 75
    if filename is None:
        prefix = os.path.basename(rampfits.filename()).split('.')[0]
        filename = f'../data/cubes/{prefix}_{datatype}.dat'        
    if slicey is None:
        shape_y = nrows
    else:
        shape_y = min(nrows, slicey.stop) - slicey.start
    if slicex is None:
        shape_x = ncols
    else:
        shape_x = min(ncols, slicex.stop) - slicex.start       
            
    start = datetime.datetime.now ()
    arr =  np.memmap ( filename, dtype='f4', mode='w+', shape=(nreads, shape_y, shape_x ))    
    #np.zeros([nreads, shape_y, shape_x])
    for ix in range(1,nreads+1):
        if timeit and (ix-1) % spacer == 0:
            print ( '[ix %i] %i sec elapsed' % (ix, ( datetime.datetime.now () - start ).seconds))
            sys.stdout.flush()
            
        if datatype == 'data':
            arr[ix-1] = rampfits[f'IMAGE_{ix}'].data[slicey,slicex]
        elif datatype == 'irp':
            arr[ix-1] = rampfits[f'REF_{ix}'].data[slicey,slicex]
        else:
            arr[ix-1] = (rampfits[f'IMAGE_{ix}'].data - rampfits[f'REF_{ix}'].data)[slicey,slicex]
            
    arr.flush ()
    del arr
    read_arr = np.memmap ( filename, dtype='f4', mode='r+', shape=(nreads, shape_y, shape_x ))
    return read_arr