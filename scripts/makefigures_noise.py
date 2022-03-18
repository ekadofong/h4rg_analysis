import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from ics.hxutils import hxramp
from h4rg_analysis import ramputils, io, plot


# \\ Plotting choices
plt.rcParams['figure.figsize'] = (6,6)
plt.rcParams['font.size'] = 14
dtypes_d = ['IRP','data','corr']
colors_d = ['grey','tab:red','b']

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
        var[ddx] = np.load(var_path)
    return var

def mk_varhists ( var, bins=None, ax=None, upper=None):
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
    ax.legend (loc='upper left', frameon=False)
    #ax.set_xscale('log')    
    return ax, stats_arr