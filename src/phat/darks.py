
import datetime
import numpy as np
from ics.hxutils import hxramp

class OLSBundle (object):
    def __init__ ( self, rate_arr, rateSE_arr, offset_arr, offsetSE_arr, unit ):
        self.rate_arr = rate_arr
        self.rateSE_arr = rateSE_arr
        self.offset_arr = offset_arr
        self.offsetSE_arr = offsetSE_arr
        self.unit = unit
    
    
class DarkRamp ( hxramp.HxRamp ):
    def ols (self, compute_indices=None, p0=0, p1=300, verbose=True, gain=None, exptime=None,
             frametype='corr'):
        if verbose:
            start = datetime.datetime.now ()
        nreads = p1 - p0
        if compute_indices is None:
            compute_indices = np.logspace(1, np.log10(nreads - 1), 10).astype(int)

        # \\ set up target arrays
        x = np.arange ( 1, nreads+1 )
        xy_sum  = np.zeros([self.nrows, self.ncols])
        y_sum   = np.zeros_like(xy_sum)
        ysq_sum = np.zeros_like(y_sum)
        rate_arr = np.zeros([len(compute_indices), self.nrows, self.ncols])
        offset_arr = np.zeros_like(rate_arr)
        
        rateSE_arr = np.zeros_like(rate_arr)
        offsetSE_arr = np.zeros_like(rate_arr)

        # \\ tally up reads
        jey = 0
        for idx in range(0, nreads):
            if verbose and (idx % 50 == 0):
                elapsed = datetime.datetime.now () - start
                print(f'IDX{idx} [{elapsed.seconds} sec]')
            if frametype == 'corr':
                frame = self.readN(idx + p0)
            elif frametype == 'data':
                frame = self.dataN(idx + p0).astype(float)
            elif frametype == 'irp':
                frame = self.irpN(idx + p0).astype(float)
            else:
                raise ValueError (f"Frame type {frametype} not recognized!")
            y_sum += frame
            xy_sum += frame * (idx+1)
            ysq_sum += frame**2
            
            if idx in compute_indices:
                stop = idx + 1
                rate, rate_SE, offset, offset_SE, unit = self._ols_rate ( x[:stop], y_sum, xy_sum, ysq_sum,
                                            gain=gain, exptime=exptime )
                rate_arr[jey] = rate
                offset_arr[jey] = offset
                rateSE_arr[jey] = rate_SE
                offsetSE_arr[jey] = offset_SE
                jey += 1
            
        if verbose:
            elapsed = datetime.datetime.now () - start
            print(f'Completed in {elapsed.seconds} sec')
            
        obundle = OLSBundle (rate_arr, rateSE_arr, offset_arr, offsetSE_arr, unit)
        obundle.interleaveOffset = self.interleaveOffset
        obundle.exptime = self.header()['W_FRMTIM']
        obundle.indices = compute_indices
        return obundle #rate_arr, rateSE_arr, offset_arr, offsetSE_arr, unit
           
           
    def _ols_rate ( self, x, y_sum, xy_sum, ysq_sum, gain=None, exptime=None):
        '''
        Compute OLS rate in either electrons/s or 
        ADU/read
        '''
        # \\ compute OLS rate 
        nreads = x.size
        ymean = y_sum / nreads
        xmean = x.mean()

        ss_xy = xy_sum - nreads * xmean * ymean
        ss_xx = np.sum(x**2) - nreads*xmean**2
        ss_yy = ysq_sum - nreads * ymean**2
        inner =  (ss_yy - ss_xy**2/ss_xx) / (nreads-2)
        s = np.sqrt ( np.where(inner>0.,inner,np.NaN) )

        if gain is not None:
            if exptime is None:
                exptime = self.header()['W_FRMTIM'] 
            unit = 'e/s'
            unit_factor = gain/exptime 
        else:
            unit = 'ADU/read'
            unit_factor = 1.
            
        rate = ss_xy/ss_xx * unit_factor    
        rate_SE = s / np.sqrt( np.where(ss_xx>0., ss_xx, np.NaN) ) * unit_factor        
        offset = ymean - ss_xy/ss_xx*xmean * unit_factor
        
        q = xmean**2 / ss_xx
        offset_SE = s * np.sqrt ( 1/nreads + np.where(q>0., q, np.NaN) ) * unit_factor
                    
        return rate, rate_SE, offset, offset_SE, unit
    
    
