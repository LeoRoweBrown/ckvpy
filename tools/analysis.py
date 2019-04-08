import numpy as np 

def calc_error(wl_range, wl, th):
    raise NotImplementedError

def _interp_angle(self, wavelength, a='default', band='0'):
        raise NotImplementedError

        """Interpolate between two points of data to find angle at desired 
        wavelength for given band e.g. wavelength=250nm -> interpolate angle
        between between points either side of 250nm
        
        Params:
        wavelength (float): desired wavelength (in m or same unit as data) at
            which to solve for angle 
        a (str): key for unit cell size in data dictionary
        """
        i = 0
        band = str(band)
        # make sure in ascending order for line 
        # "while float(wl[i]) < wavelength"
        wl = self.data[a][band]['wavelength']
        th = self.data[a][band]['angle']
        
        while float(wl[i]) < wavelength:
            i += 1 # condition stops when data point is greater than
                   # wavelength, so look at i and i-1 for correct range
            if i > len(wl):
                raise ValueError("Failed to find angles for wavelength =",
                                 wavelength,
                                 ". Check that it exists in data.")
        if wl[i-1] > wavelength: 
            # wl[i-1] should be the value smaller than desired wavelength
            print("Wavelength too small! Changing", wavelength, "to", wl[i-1])
            print("Dataset doesn't seem to match desired wavelength range, "
                  "expect strange results (it is likely that the angles "
                  "found will not be anywhere near the vertical line")
            wavelength = wl[i-1]
        elif wl[i] < wavelength: 
            # wl[i] should be the value larger than desired wavelength
            print("Wavelength too large! Changing", wavelength, "to", wl[i])
            raise Warning("Dataset doesn't seem to match desired wavelength "
                          "range, expect strange results (it is likely that "
                          "the angles found will not be anywhere near the "
                          "vertical line")
            wavelength = wl[i]

        if (wl[i]-wl[i-1]) < 1.e-15*(th[i]-th[i-1]): # avoid division by zero
            angle = (th[i]+th[i-1])/2 # take average if angle is multivalued
                                      # if wl1 and wl2 are the same
        else:
            angle = (th[i]-th[i-1])/(wl[i]-wl[i-1])*(wavelength-wl[i-1]) \
                + th[i-1] # grad*(wavelength1-wavelength0) + angle0
        print("found", (wavelength,angle), "between", (wl[i], th[i]), \
              "and", (wl[i-1], th[i-1]) )
        return wavelength, angle, i

def wl_cut(self, wl, param, wl_range=[0.,1e10], sign=1):
    """Take cut of data based on wavelength range. Default behaviour
    removes negative angles"""
    wl_nm_range = []
    theta_nm_range = []
    param_nm_range = []
    for i, w in enumerate(wl):
        if w < wl_range[1] and w > wl_range[0] and sign*theta[i]>0:
            wl_nm_range.append(w)
            theta_nm_range.append(theta[i])
            if param is not None:
                param_nm_range.append(param[i])
    if param is None:
        return theta_nm_range, wl_nm_range
    else:
        return theta_nm_range, param_nm_range, wl_nm_range