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
        fp[r_i] = irpImage[slicey,slicex]
    fp.flush ()
    return fp, filename

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
        fp[r_i] = dataImage[slicey,slicex]
    fp.flush ()
    return fp, filename


def corrStack_memmap ( ramp, r0=0, r1=-1, slicey=None, slicex=None, filename=None):
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
        '''
        #n = ramp._readIdxToFITSIdx ( r_i )
        #data_extname = f'IMAGE_{n}'
        #dataImage = ramp.dataN(r_i) #fits[data_extname].read ()
        
        try:
            #ref_extname = f'REF_{n}'
            #irpImage = ramp.fits[ref_extname].read ()
            #irpImage = hxramp.constructFullIrp(irpImage, ramp.nchan, refPix=ramp.interleaveOffset)
            irpImage = ramp.irpN(r_i)
            corrImage = dataImage[slicey,slicex].astype(float) - irpImage[slicey,slicex].astype(float)
        except: # \\ no IRP frame
            print('no IRP frame! (r_i=%i)' % r_i)
            corrImage = dataImage[slicey,slicex].astype(float)
        '''
        corrImage = ramp.readN(r_i)[slicey,slicex] 
        fp[r_i] = corrImage
    fp.flush ()
    return fp, filename