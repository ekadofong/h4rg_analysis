import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import stats
from ics.hxutils import hxramp
from h4rg_analysis import ramputils, io, plot


# \\ Plotting choices
plt.rcParams['figure.figsize'] = (6,6)
plt.rcParams['font.size'] = 14
dtypes_d = ['IRP','data','corr']
colors_d = ['#666666','tab:red','b']

# \\ lab-decided values
bestbaselines = open ('../data/paths/best_ever_baseline_darks_8_21_2022.txt', 'r').read().splitlines()[1:]
gain = 3.2 # e-/ADU, measured by JHU for detector 18660

def load_rampids ( jhu_paths=None ):
    if jhu_paths is None:
        jhu_paths = bestbaselines
    ramp_ids =  np.array([ os.path.basename(io.jhu2tiger(bx)).split('.')[0] for bx in bestbaselines ])
    return ramp_ids

def load_variancemaps ( ramp_id, Ny=4096, Nx=4096 ):
    '''
    Load computed per-pixel variance maps for each of the 
    '''
    var = np.zeros([3, Ny, Nx])
    for ddx in range(3):
        data_type = dtypes_d[ddx]
        
        var_path = f'../data/output/{ramp_id}_{data_type}_S0-E-1_var.npy'    
        if os.path.exists(var_path):
            var[ddx] = np.load(var_path)
        else:
            var[ddx] = np.NaN
    return var

def mk_varhists ( var, bins=None, ax=None, upper=None, show_legend=True):
    if bins is None:
        bins = np.linspace(10.,60.,300)
    if ax is None:
        ax = plt.subplot(111)
    if isinstance(upper, float):
        upper = (upper,upper,upper)

    stats_arr = np.zeros([var.shape[0], 6])
    for ddx in range(3):
        if upper is None:
            vclipout = stats.sigmaclip(var[ddx])
            var_sclip = np.sqrt(vclipout.clipped) * gain
            stats_arr[ddx, 5] = vclipout.upper
        else:
            var_sclip = np.sqrt(var[ddx][var[ddx]<upper[ddx]])*gain
            stats_arr[ddx, 5] = upper[ddx]
            
        stats_arr[ddx, 0 ] = np.mean(var_sclip)
        stats_arr[ddx, 1 ] = np.std(var_sclip)
        stats_arr[ddx, 2:5 ] = np.quantile(var_sclip, [.16,.5,.84])    
        out=ax.hist( var_sclip, bins=bins, label=dtypes_d[ddx], color=colors_d[ddx], histtype='step', lw=3 )
        #ax.hist( var[ddx].flatten()*gain, bins=bins, color=colors_d[ddx], 
        #        histtype='step', ls=':' ) #\\ also show un-sigclipped variance for completeness' sake
        nmetric = stats_arr[ddx,3]
        ax.axvline ( nmetric, color=plot.adjust_value(colors_d[ddx]), ls='--')
        ax.annotate ( f'  {nmetric:.0f} e', (nmetric,.9*out[0].max()), color=colors_d[ddx] )
        
        
        
    ax.set_xlabel ( r'$\sigma$ (e)')
    ax.set_ylabel ('N')
    if show_legend:
        for ddx in range(3):
            ax.text ( 0.025, 0.95 - ddx*.075, dtypes_d[ddx], color=colors_d[ddx], va='top',
                     transform=ax.transAxes )
    #ax.set_xscale('log')    
    return ax, stats_arr

def load_ratemap ( ramp_id, readtime=None, ax=None ):
    '''
    Load estimated up-the-ramp rate in e / sec
    '''
    utr_path = f'../data/output/{ramp_id}_corr_S0-E-1_utrparam.npy' 
    utrparam = np.load(utr_path)
    if readtime is None:
        readtime = 10.85705 # \\ XXX seconds, default for bestbaselines (ramp.header  ()['W_FRMTIM'])
    measrate_e = utrparam[0] / readtime  * gain
    return measrate_e

def mk_utrrates ( measrate_e, ax=None ):
    if ax is None:
        ax = plt.subplot(111)
        
    im=ax.imshow(measrate_e, vmin=1e-3, vmax=1., norm=colors.LogNorm() )
    plt.colorbar(im, ax=ax, label=r'R (e/s)')