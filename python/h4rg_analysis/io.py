import tempfile
import numpy as np
from ics.hxutils import hxramp

def jhu2tiger ( pathtofile ):
    jhu_root = "/data/"
    tiger_root = "/projects/HSC/PFS/JHU/"
    return tiger_root + pathtofile.strip(jhu_root).replace('83.fits', '23.fits')

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

def irpStack_memmap ( ramp, r0=0, r1=-1, filename=None ):
    if filename is None:        
        filename = tempfile.NamedTemporaryFile()
        
    r0 = ramp._readIdxToAbsoluteIdx ( r0 )
    r1 = ramp._readIdxToAbsoluteIdx ( r1 )    
    fp = np.memmap ( filename, dtype='u2', mode='w+', shape=(r1-r0+1, ramp.nrows, ramp.ncols ))
    for r_i in np.arange(r0, r1+1):
        n = ramp._readIdxToFITSIdx ( r_i )
        extname = f'REF_{n}'
        irpImage = ramp.fits[extname].read ()
        irpImage = hxramp.constructFullIrp(irpImage, ramp.nchan, refPix=ramp.interleaveOffset)
        fp[r_i] = irpImage
    fp.flush ()
    return fp, filename

def dataStack_memmap ( ramp, r0=0, r1=-1, filename=None ):
    if filename is None:
        filename = tempfile.NamedTemporaryFile()
        
    r0 = ramp._readIdxToAbsoluteIdx ( r0 )
    r1 = ramp._readIdxToAbsoluteIdx ( r1 )
    fp = np.memmap ( filename, dtype='u2', mode='w+', shape=(r1-r0+1, ramp.nrows, ramp.ncols ))    
    for r_i in np.arange(r0, r1+1):
        n = ramp._readIdxToFITSIdx ( r_i )
        extname = f'IMAGE_{n}'
        dataImage = ramp.fits[extname].read ()
        fp[r_i] = dataImage
    fp.flush ()
    return fp, filename


def corrStack_memmap ( ramp, r0=0, r1=-1, filename=None):
    if filename is None:
        filename = tempfile.NamedTemporaryFile()
        
    r0 = ramp._readIdxToAbsoluteIdx ( r0 )
    r1 = ramp._readIdxToAbsoluteIdx ( r1 )
    fp = np.memmap ( filename, dtype='u2', mode='w+', shape=(r1-r0+1, ramp.nrows, ramp.ncols ))        
    
    for r_i in np.arange(r0, r1+1):
        n = ramp._readIdxToFITSIdx ( r_i )
        data_extname = f'IMAGE_{n}'
        dataImage = ramp.fits[data_extname].read ()
        
        try:
            ref_extname = f'REF_{n}'
            irpImage = ramp.fits[ref_extname].read ()
            irpImage = hxramp.constructFullIrp(irpImage, ramp.nchan, refPix=ramp.interleaveOffset)
            corrImage = dataImage - irpImage
        except: # \\ no IRP frame
            corrImage = dataImage
        
        fp[r_i] = corrImage
    fp.flush ()
    return fp, filename