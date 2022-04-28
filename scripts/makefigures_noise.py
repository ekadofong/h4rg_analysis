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
    
def mk_statarrviz (statarr, Nreads, expected_dc=None ):
    fig, axarr = plt.subplots(1,2,figsize=(12,5))
    ax = axarr[1]
    for n in np.random.randint(0, statarr[0,0].size, 100):
        ypix, xpix = np.unravel_index ( n, statarr.shape[2:] )
        ax.plot(Nreads, statarr[1,:,ypix,xpix] / np.sqrt(Nreads), color='grey', alpha=0.3 )
        axarr[0].plot(Nreads, statarr[0,:,ypix,xpix], color='grey', alpha=0.3)
        

    xs = np.logspace(np.log10(Nreads.min()), np.log10(Nreads.max()), 45)
    mstart = np.nanmean(statarr[1,0]/np.sqrt(Nreads[0]))
    ys = 1./np.sqrt(xs)
    ys *= mstart / ys[0]
    ax.plot(xs,ys, color='k', lw=3, ls='--')
    if expected_dc is not None:
        axarr[0].axhline(expected_dc, color='k', ls='--', lw=3)

    mm =  np.nanmean(statarr[0],axis=(1,2))
    msd = np.nanmean(statarr[1],axis=(1,2))
    sd_mm = np.nanstd(statarr[0],axis=(1,2))
    ax.plot ( Nreads, msd / np.sqrt(Nreads), color='lime', lw=3 )
    axarr[0].plot(Nreads, mm, color='r', lw=3 )

    #ax.axhline(np.interp(100, Nreads, msd)/np.sqrt(100), color='r', ls='--')


    axarr[0].plot ( Nreads, msd/np.sqrt(Nreads), color='lime')
    axarr[0].plot ( Nreads, -msd/np.sqrt(Nreads), color='lime')
    axarr[0].plot ( Nreads, sd_mm, color='r')
    axarr[0].plot ( Nreads, -sd_mm, color='r')
    axarr[1].plot ( Nreads, sd_mm, color='r', )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(xs.min(),xs.max())
    axarr[0].set_xlim(xs.min(),xs.max())
    ax.set_xlabel(r'N$_{\rm reads}$')
    ax.set_ylabel(r'SE($R$) (e/s)')
    axarr[0].set_ylim(-1.,1.)
    axarr[1].set_ylim(0.01, 2.)
    axarr[0].set_xlabel(ax.get_xlabel())
    axarr[0].set_ylabel(r'$\bar R$ (e/s)')
    plt.tight_layout ()
    
def print_r2rstats ( rateA, rdiff, exptime, nreads, gain ):
    unit_conversion = gain / exptime
    clipped_rate = stats.sigmaclip(rateA)
    shot_noise = np.sqrt(abs(np.median(clipped_rate.clipped * unit_conversion)) * exptime * nreads)
    print('Median rate: %.4f e/s' % np.median(clipped_rate.clipped * unit_conversion))

    rdiff_phys = rdiff * unit_conversion
    sclip = stats.sigmaclip(rdiff_phys)
    clipped =sclip.clipped
    rms = np.sqrt ( np.sum(clipped**2) / (2.*float(clipped.size)))

    totrms  = rms * exptime * nreads
    print('Expected shot noise: %i e' % shot_noise)
    print('Observed total RMS: %i e' % totrms)
    print('SQRT[RMS^2 - shot^2] = %.2f e' % np.sqrt(totrms**2 - shot_noise**2))
        

def show_r2rdiff ( rateA, rateB, exptime, nreads, gain, colsub=1. ):
    '''
    Show a map of the difference in estimated signal (in e/s) for
    two ramps, and the impact of removing a columnated structure of width
    colsub pixels from the residual
    '''
    
    unit_conversion = gain/exptime
    
    # \\ figure to show rates    
    fig0, axarr = plt.subplots(1,2,figsize=(21*2/3,5))
    plot.scaled_imshow ( rateA * unit_conversion, ax=axarr[0], label=r'$\rm R$ (e/s)' )
    plot.scaled_imshow ( rateB * unit_conversion, ax=axarr[1],  label=r'$\rm R$ (e/s)' )    
    plt.tight_layout ()
    #plt.savefig('../figures/20220421_corr-medsub.png')        
    
    # \\ figure to show rate difference
    rdiff = rateA - rateB
    print_r2rstats ( rateA, rdiff, exptime, nreads, gain )
    rdiff_mc,colmed_rdiff = ramputils.colsub ( rdiff, colsub )
    
    fig1, axarr = plt.subplots(1,3,figsize=(21,5))
    plot.scaled_imshow ( rdiff * unit_conversion , ax=axarr[0], label=r'$\Delta \rm R$ (e/s)' )
    plot.scaled_imshow ( (rdiff - rdiff_mc)*unit_conversion, ax=axarr[1],  label=r'$\Delta \rm R$ (e/s)' )
    plot.scaled_imshow ( rdiff_mc * unit_conversion, ax=axarr[2],  label=r'$\Delta \rm R$ (e/s)')

    for rd, ax in zip([rdiff, rdiff_mc], axarr[[0,2]]):
        rdiff_phys = rd * unit_conversion
        clipped = stats.sigmaclip(rdiff_phys).clipped
        rms = np.sqrt ( np.sum(clipped**2) / (2.*float(clipped.size)))
        ax.text ( 0.025, 0.975, r'$\rm RMS_{\rm clpd}=%.4f$ e/s'%rms, color='w', 
                        transform=ax.transAxes,
                        fontsize=18,
                        ha='left', va='top', )
        
    axarr[0].set_title ( 'Rate Difference map (residual)' )
    axarr[1].set_title ( 'Columnated model' )
    axarr[2].set_title ( 'residual - model')

    plt.tight_layout ()
    #plt.savefig('../figures/20220421_corr-medsub.png')    
    return fig0, fig1