import numpy as np 
import csv
from os import listdir
from scipy import interpolate
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from bases import CSVLoader

class Intersect3D(bases.CSVLoader):
    """Intersect 2D band structure with electron plane. Can be used for 3D data or 2D data, but 3D data must be confined to a 2D
    plane which at the moment is constrained to just planes in k_rho,kz where k_rho = norm(kx,ky). Realistically in COMSOl this
    means only kz,kx and kz,ky because of the way bands are found - choose a direction from |k|=0 and increase |k|. Always sorted
    by band (band is the root key)""" 
    def __init__(self, datafile, headers=['skip', 'band', 'skip' 'frequency', 'kx', 'ky', 'kz', 'n', 'skip'],\
        sort_by = 'band', ndim=3):

        plt.rcParams.update({'font.size': 14})
        np.set_printoptions(precision=3) # will this affect data saved to text?
        self.ndim = ndim
        for header in headers:
            if header not in ['band', 'frequency', 'kx', 'ky', 'kz', 'n', 'skip']:
                raise ValueError("Invalid header supplied, must be one of ['band', 'frequency', 'kx', 'ky', 'kz', 'n']")
        return super(bases.Bzone2D, self).__init__(datafile, param=param, headers=headers, ndim=ndim)


    def interpolate_surface(self, resolution=0.01):
        """Range is -0.5 to 0.5 so resolution must be multiple of 10 + 1? Resolution of longest dimension"""
        # first bin the data
        np.max(self.data[band]['k_rho'])
        np.max(self.data[band]['kz'])
        noelements_rho = int(1/resolution + 1)
        noelements_z = int(1/resolution + 1)
        kx = np.linspace(-0.5,0.5,noelements)
        ky = np.linspace(-0.5,0.5,noelements)

        my, mx = np.meshgrid(kx,ky)
        my = np.copy(my[::-1])
        mf = { keys: np.zeros_like(mx) for keys in self.data } # list of matrices for each band
        interpolated = { keys: np.zeros_like(mx) for keys in self.data }
        # convert polar coords into cartesian
        for band in self.data:
            x_list = [None]*len(self.data[band]['k_theta'])
            y_list = [None]*len(x_list)

            i = 0 # iterator for angle/radius/freq
            xs = []
            for i, angle in enumerate(self.data[band]['k_theta']): # could also loop over k_abs
                #f = self.data['f']
                r = self.data[band]['k_abs'][i]
                y = r*np.sin(angle)
                x = r*np.cos(angle)

                x_list[i] = x
                y_list[i] = y

                # now bin the data
                col_index = int(np.round((x+0.5)/resolution)) # redundant for interp
                row_index = int(np.round((-y+0.5)/resolution))
                
                # frequency matrix
                mf[band][row_index, col_index] = self.data[band]['f'][i] # redundant for interp
                f = self.data[band]['f']

            # interpolate griddata takes care of binning
            #print(len(x_list), len(y_list), len(mf[band]))
            # outside of convex hull the values are NaN, so replace these with nearest neighbour
            interpolate_hull = interpolate.griddata((np.array(x_list), np.array(y_list)), np.array(f), (mx,my), method='cubic') # aliasing issues when we bin again?
            fill_matrix = interpolate.griddata((np.array(x_list), np.array(y_list)), np.array(f), (mx,my), method='nearest')
            # calculate mask based on NaN (interpolate!=interpolate gives logic matrix where NaNs evaluate to true (1))
            #mask = np.ones_like(interpolate)*(interpolate_hull!=interpolate_hull)
            
            interpolate_hull[np.isnan(interpolate_hull)] = 0
            mask = np.ones_like(interpolate_hull) * (interpolate_hull==0)
            #print(mask)
            #print(fill_matrix)
            interpolated[band] = mask*fill_matrix + interpolate_hull*(mask==0) # invert mask with mask==0
            #print(interpolate_hull*(mask==0))

        print(interpolated['0'])
        #print(self.data['4']['f'][0]) # being overwritten by band 5
        fig = plt.figure(figsize=(12,9))
        #ax = fig.gca(projection='3d')
        for i, key in enumerate(interpolated):
            ax = fig.add_subplot(2,3,i+1, projection='3d')
            ax.set_title("Band "+str(i+1)+" 2D dispersion")
            ax.set_xlabel("kx")
            ax.set_ylabel("ky")
            #print((interpolated[key][0,0]))

            surf = ax.plot_surface(mx, my, interpolated[key], cmap=cm.bwr,
                            linewidth=0, antialiased=False)
            plane = ax.plot_surface(mx, my, my*0.9*3.e8*np.pi/a, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
            ax.set_zlim([np.min(interpolated[key]),np.max(interpolated[key])])
        #plt.show()
        return mx, my, interpolated

    def intersect(dispersion, grid_x, grid_y, v=0.9):
        """course intersection, then fine by looking in neighbourhood. Do this by interpolating between lines?"""

        electron_plane = grid_y*v
        #e_line = electron_plane[0]
        grid_size = grid_x.shape[0]
        crossings = [[None]*50]*3
        crossings = { str(band_name): {'kx': [], 'ky': [], 'f': []} for band_name in range(len(dispersion))}
        # we are working to intersect in y direction, so tranpose matrices
        #grid_x_t = grid_x.T
        #grid_y_t = grid_y.T
        # isntead  of using grid, make simple ranges 
        x_array =  np.arange(-0.5, 0.5, 0.02)
        y_array = np.arange(-0.5, 0.5, 0.02)
        #dispersion = dispersion.T
        step = int((grid_size-1)/50) # coarse is 51x51 electron plane 
        for band in dispersion:
            #disp = np.copy(dispersion[band].T)
            disp = dispersion[band].T
            for xi, x in enumerate(x_array[:-1]):
                #print(x, xi)
                for yi, y in enumerate(y_array[:-1]):
                    #print(y, yi)
                    y_next = y_array[yi + 1]
                    f = disp[xi*step, yi*step]
                    f_next = disp[xi*step, yi*step + step]
                    ve = v*3.e8*2*np.pi/a # 2pi/a and speed of light
                    m = (f_next-f)/(y_next-y)
                    #y_cross = y*ve # ve*k = f
                    if abs(m) < 1e-50:
                        #print('skipping')
                        y_cross = f/ve
                    else:
                        c = f - m*y
                        y_cross = c/(ve - m)
                    #y_cross = f/(v*3.e8*np.pi-(f_next-f)/(y_next-y)) # crossing point of lines, potential for div 0 error
                    #print(y_cross*ve-f, f_next-y_cross*ve)
                    #print(y_cross)

                    if abs(y_cross) <= 0.5 and y_cross*ve >= f and y_cross*ve <= f_next:
                        #print(y)
                    #    print('crossing at y=', y_cross,'x=',x,'f=',y_cross*ve)
                        crossings[band]['kx'].append(x) # replace with intialised lists for speed?
                        crossings[band]['ky'].append(y_cross)
                        crossings[band]['f'].append(y_cross*ve)
                        #print(dispersion[band][step*yi,step*xi])
        # cols paramterise y, rows parameterise x

        kx = np.array(crossings['2']['kx'])
        ky = np.array(crossings['2']['ky'])
        f = np.array(crossings['2']['f'])
        #print(crossings['2']['ky'])
        #print(crossings['2']['kx'])
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='3d')
        ax.plot(xs=crossings['2']['kx'], ys=crossings['2']['ky'], zs=crossings['2']['f'], label='Intersection')
        ax.set_xlabel("kx")
        ax.set_ylabel("ky")
        ax.set_xlim([-0.5,0.5])
        ax.set_ylim([-0.5,0.5])
        #plane = ax.plot_surface(grid_x, grid_y, grid_y*ve, cmap=cm.coolwarm,
        #                    linewidth=0, antialiased=False)
        #print(np.shape(dispersion['0']))
        #print(type(dispersion['0']))
        #print(type(grid_x))
        #print(np.shape(grid_x))
        #print(np.shape(grid_x))
        #ax.plot_surface(grid_x, grid_y, dispersion['0'], cmap=cm.bwr,
                        #linewidth=0, antialiased=False)
        plt.show()
        wl = 2*np.pi*3.e8/f
        theta = np.arctan(kx/ky)
        cher = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_title('cherenkov')
        ax.plot(theta, wl)

        plt.show()


    def replacenans(self):
            """Replace NaNs and replace with extrapolation of key_y data based on key_x data as the independent variable,
            i.e. delta(y)/delta(x) gives gradient which is used to extrapolate."""
            for root in self.data:
                y = self.data[root][key_y]
                x = self.data[root][key_x]
                for band in data:
                    i = 1
                    isnan = False
                    reverse = 0
                    while(not math.isnan(band[i])):
                        if reverse > 1:
                            raise ValueError("Already attempted to reverse data, cannot interpolate NaNs automatically.")
                        j = 0
                        while (x[i]-x[i-1]) < 1.e-15*(y[i]-y[i-1] and i-j > 0): # deal with divide by zero
                            grad = (y[i]-y[i-j])/(x[i]-x[i-j]) # find gradient between two points further and further apart
                            j += 1 # until reach beginning of data
                            if i-j < 0: 
                                raise Warning("""Unable to interpolate (due to gradient) for NaN replacement, trying to
                                switch axes""")
                                reverse += 1
                                self.sort_data(key_y, direction=-1)

                        grad = (y[i]-y[i-1])/(x[i]-x[i-1])
                        if math.isnan(grad):
                            raise Warning("""gradient for interpolation is NaN, data probably starts with NaN, trying to
                            call sort_data(<key>, direction=-1) to reverse order""")
                            self.sort_data(key_y, direction=-1)
                            reverse += 1
                        i += 1
                        isnan =  math.isnan(band[i])
                    # now we replace the NaN at index i
                    self.data[root][key_y][i] = y[i-1] + grad*(x[i]-x[i-1])
