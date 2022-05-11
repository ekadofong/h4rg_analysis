import numpy as np
from scipy import stats


def sigclip_insitu ( array, low=3., high=3., ):
    clipped = stats.sigmaclip(array[np.isfinite(array)], low=low, high=high)
    cl_array = np.where((array<clipped.upper)&(array>=clipped.lower), array, np.NaN)
    return cl_array