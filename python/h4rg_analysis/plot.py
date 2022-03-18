import numpy as np
from matplotlib import colors

def adjust_value ( color, factors=(1.,1.,0.8) ):
    if isinstance(color, str):
        color = colors.to_rgb ( color )
    chsv = colors.rgb_to_hsv ( color )
    chsv = chsv * np.array(factors)
    crgb = colors.hsv_to_rgb ( chsv )  
    return crgb  
