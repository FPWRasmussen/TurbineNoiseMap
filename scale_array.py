import numpy as np
from scipy.interpolate import RectBivariateSpline


def scale_array(array, scaling_factor):
        shape = np.asarray(np.shape(array))
        new_shape = np.round(shape * scaling_factor).astype(int)
        
        # scaled_array = np.zeros(new_shape)
        x = np.linspace(0, 1, shape[0])
        y = np.linspace(0, 1, shape[1])
        interp_spline = RectBivariateSpline(x, y, array)
            
        x2 = np.linspace(0, 1, new_shape[0])   
        y2 = np.linspace(0, 1, new_shape[1])
        
        scaled_array = interp_spline(x2, y2)
        return scaled_array