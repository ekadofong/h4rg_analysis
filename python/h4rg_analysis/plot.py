import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def adjust_value ( color, factors=(1.,1.,0.8) ):
    if isinstance(color, str):
        color = colors.to_rgb ( color )
    chsv = colors.rgb_to_hsv ( color )
    chsv = chsv * np.array(factors)
    crgb = colors.hsv_to_rgb ( chsv )  
    return crgb  

def scaled_imshow ( im, ax=None, beta=0.01, colorbar=True, interpolation=None ):
    if ax is None:
        ax = plt.subplot(111)
    vmin, vmax = np.nanquantile ( im, [beta, 1. - beta] )
    if interpolation is not None:
        imout = ax.imshow ( im, vmin=vmin, vmax=vmax, interpolation=interpolation )
    else:
        imout = ax.imshow ( im, vmin=vmin, vmax=vmax)
    if colorbar:
        plt.colorbar ( imout, ax=ax )
    return ax