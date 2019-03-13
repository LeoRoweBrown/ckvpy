import csv
import math
import numpy as np
import json

class CSVLoader(object):
    def __init__(self, datafile, file_suffix='undefined', headers=[], sort_by='undefined'):
        """
        Base class to dump and analyse Cerenkov data tables from COMSOL. Aim is to be adaptable to different datasets.

        Params:
        datafile (str): path of raw csv data
        
        Attributes:
        file_suffix (list<str/int>): Split mode: list of data file paramters that there is data for e.g. for when there
            is data separated into different files for different values of 'a'. The datafile supplied is the prefix i.e. foo_ for
            foo_100 *without* the .csv extension. Will be automatically read and sorted in the data dictionary.
            Single mode: Has length=1, contains string of paramter value that is not stored in data for supplied data file, but is
            used as a key for the data dicitonary. E.g. data[a], for data being sorted by different unit cells.
        
        prefix (str): only used when looping over file_suffix in split mode. Prefix to filename (see attribute file_suffix).

        format (str): format of the data: 'split' means separate files for each file_suffix paramter and is for automatically looping 
            over unit cell size, 'single' means only one parameter is being considered and 'merged' means data for different 
            paramters is in one file. This is automatically determined.
        
        data (dict[key = paramter: value = list[float] ]): Data dictionary sorted by parameter value as the key, same structure 
            as datafile.
        path (str): path of datafile, set in read(), used for automatic graph titles if no model name supplied
        col (dict[key = variable name : column index]): dictionary of column numbers for each variable - for dealing with differently
            formatted data tables
        headers (list<str>): list of paramters featured in data table in order of appearance (left to right). Use 'skip' to ignore
            a column. Example: ['band', 'k', 'wavelength', 'angle', 'a'].
        sort_by (str): paramter in headers by which the data dictionary is sorted by - sort_by determines the root key. That is the
            outermost index i.e. if sort_by = 'a', data = { 100nm : { <data> } 200nm : { <data> } ... }.
        """

        self.prefix = None # only used when looping over 'file_suffix' to find angle against unit cell size
        self.data = {} # dictionary of data, keys are unit cell size
        self.path = None
        self.file_suffix = []
        self.sort_by = sort_by
        self.col = {}
        self.num_bands = {}
        
        for n, header in enumerate(headers):
            if header is 'skip':
                continue
            self.col[header] = n

        if type(file_suffix) is list and file_suffix[0] is not 'undefined':
            self.format = 'split'
            self.prefix = datafile
            print("List of unit cell sizes has been supplied. These will used in find_angle automatically unless loop=False.")
            print("""NOTE: for the entire dataset (wavelength vs angle), you must choose desired value of 'file_suffix'\
to read by the method read(<path>). Otherwise, the first value in the list will be used.""")
            datafile += str(file_suffix[0]) + ".csv"
        elif type(file_suffix) is str or file_suffix[0] is 'undefined':
            self.file_suffix = [file_suffix]
            self.format = 'single'
        try:
            self.read(datafile) # use first value in list by default
            if len(self.file_suffix) > 0:
                self.format = 'merged'
        except FileNotFoundError as f:
            raise f
        except OSError:
            raise OSError("Failed to parse filename, ensure format: '..\\dir\\subdir\\file[.csv]' or '../dir/subdir/file[.csv]'")
        except:
            raise IOError(str(
                "Failed to load raw data into dictionary, check the format. The variables stored in the dictionary can be"
                "changed using headers = ['col1', 'col2' ...] where the elements are table headings."))
# fix this indentation mess.. f strings only supported in python3
        
    def _get_raw(self, path):
        with open(path) as file:
            print("Reading file", path)
            reader = csv.reader(file)
            for n_rows, row in enumerate(reader):
                if '%' in ''.join(row): # skip headings etc.
                    print("ommitting", row)
                    continue
                #if self.format is 'merged' or self.format is 'single':
                try:
                    root = row[ self.col[self.sort_by] ] # root key i.e. self.data[root]
                except KeyError:
                    raise KeyError("Chosen sort_by key does not exist in list of headers for this data set")
                except IndexError:
                    print("""Using supplied value for root key from 'file_suffix' since it is not found in file, or\
                        table headings are formatted incorrectly""")
                    root = self.file_suffix[0]
                
                if root not in self.data: # for repeated calls of _get_raw with different root keys e.g. for a, '1.00E-07', '1.50E-07'
                    self.data[root] = {}
                    self.data[root]['raw'] = []
                self.data[root]['raw'].append([float(complex(col.replace('i','j')).real) for col in row]) # short CSV, just append
        return n_rows

    def read(self, path):
        """Open CSV and dump into list of lists, 
        [band number, k, wavelength, Cherenkov angle]
        
        Params:
        Path (str): file path of CSV file
        """
        self.path = path
        # split into bands, etc.
        #print("preloop, param =", self.file_suffix)
        for suffix in self.file_suffix: # for split mode
            if self.format is 'split':
                path = self.prefix + str(suffix) + ".csv"
            n_rows = self._get_raw(path)
            if self.format is 'single':
                break # not in split mode so only need one iteration

        for root in self.data.keys(): # fill each entry according to root key
            num_bands = int(max(self.data[root]['raw'][ self.col['band']] ))
            self.num_bands[root] = num_bands
                
            for key in self.col:
                # avoid storing param key twice 
                # e.g. avoid data['1.00E-07']['1.00E-07']

                if key is self.sort_by: 
                    continue
                # split data into bands (sub lists)
                self.data[root][key] = [None]*num_bands 
                
            for b in range(num_bands):
                for key in self.col:
                    if key is self.sort_by:
                        continue
                    self.data[root][key][b] = \
                    [ data[self.col[key]] for data in self.data[root]['raw']\
                     if int(data[self.col['band']]) == b+1 ]
            self.data[root].pop('raw')


    def sort_data(self, key, direction=1):
        """sort ascending according to key, e.g. key = 'wavelength'"""
        for root in self.data:
            for b in range(self.num_bands[root]):
                data = self.data[root]
                sort_index = np.argsort(data[key][b])
                if direction == -1:
                    sort_index = sort_index[:-1]

                for param in self.data[root]:
                    self.data[root][param][b] = \
                    [data[param][b][ind] for ind in sort_index]

    def save_data(self, name):
        with open(name, 'w') as f:
            json.dump(self.data, f)

#=============================================================================#

class Bzone2D(CSVLoader):
    
    def __init__(self, datafile, headers=['skip', 'band', 'skip' 'frequency', 'kx', 'ky', 'kz', 'n', 'skip'],\
        sort_by = 'band', ndim=3):
        
        plt.rcParams.update({'font.size': 14})
        np.set_printoptions(precision=3) # will this affect data saved to text?
        self.ndim = ndim
        for header in headers:
            if header not in ['band', 'frequency', 'kx', 'ky', 'kz', 'n', 'skip']:
                raise ValueError("Invalid header supplied, must be one of ['band', 'frequency', 'kx', 'ky', 'kz', 'n']")
        return super(CSVLoader, self).__init__(datafile, param=param, headers=headers)

    def _convert_to_polar(self, inverse=False):
        for band in self.data:
            kx = np.array(self.data[band]['kx'])
            ky = np.array(self.data[band]['ky'])
            if self.ndim = 3:
                kz = np.array(self.data[band]['kz'])
                k_theta = np.arctan((kx*kx + ky*ky)**0.5/kz) # handle pi case?
                if kx < ky*1e-15: # check for kx = 0 
                    k_phi = np.pi/2
                else:
                    k_phi = np.mean(np.arctan(ky/kx)) # angle of plane our data lies on to kx axis in 3D space
                    if np.std(np.arctan(ky/kx)) > 1e-10*k_pi:
                        raise ValueError("Data must be in a form where the BZ plane is defined by xz or yz (or (x,y)z with y/x fixed) ")
            else:
                kz = np.zeros_like(ky)
                k_theta = np.arctan(ky/kx)
                k_phi = 0.0
            k_abs = kx*kx + ky*ky + kz*kz
            k_rho = kx*kx + ky*ky # in plane component
           
            self.data[band]['k_abs'] = k_abs
            self.data[band]['k_theta'] = k_theta
            self.data[band]['k_rho'] = k_rho
            self.data[band]['k_phi'] = k_phi
           

    def reflect(self, symmetry=4):
        """
        Use symmetry of irreducible Brillouin zone to reproduce full Brillouin zone. For 3D model, only valid for kx=0 or
        ky = 0 planes.
    
        Params:
        symmetry (int): Degree of symmetry (of Dihedral group D_n, n=symmetry)
        start_angle (int) Angle (DEGREES) in BZ that data starts at in degrees, assumed to be minimum angle
        """
        # copy data and make angles negative for 2nd octant

        # transformations of angle, first number is multiplier, second is addition
        reflect_th_0 = [-1.0, 0.0] # reflect in theta = 0 
        reflect_th_min45 = [-1.0, -90.0*np.pi/180.0] # reflect in theta = -45 degrees
        reflect_th_45 = [-1.0, 90.0*np.pi/180.0] # reflect in theta = 45 degrees
        reflect_th_90 = [-1.0, np.pi] #  reflect in theta = 90 degrees # will cause angles above 180,
        # but sin(180+theta) = sin(-180+theta)

        if symmetry == 4:
            reflections = [reflect_th_0, reflect_th_90]
        elif symmetry == 8
            reflections = [reflect_th_0, reflect_th_min45, reflect_th_45]
        elif symmetry == 6 or symmetry == 12:
            raise NotImplementedError(
                "Symmetry of order multiple of 6 not yet implemented"
                "e.g. for the hexagonal unit cell/reciprocal lattice"
                )
        else:
            raise ValueError("Please pick symmetry order 4 or 8")
        #data = self.data[band]
        

        for band in self.data:
            start_angle = np.min(self.data[band]['k_theta']) #only look in band 1, angle is same for all bands
            if abs(start_angle) > 1e-10:
                raise ValueError("Please only use data that starts from 0 degrees")
            self._convert_to_polar()
            #print(band)
            for params in reflections:
                new_angles = params[0]*np.array(self.data[band]['k_theta']) + params[1]
                #plane_angle = np.average(self.data[band]['plane_angle'])
                self.data[band]['k_abs'] = np.tile(self.data[band]['k_abs'], 2) # tile copies array and appends
                self.data[band]['k_theta'] = np.concatenate((self.data[band]['k_theta'], new_angles.tolist()) )
                self.data[band]['f'] += np.tile(self.data[band]['f'], 2)
                
            # now use full k_abs and k_theta data to derive kx, ky kz
            k_rho = np.array(self.data[band]['k_abs'])*np.sin(self.data[band]['k_theta'])
            self.data[band]['k_rho'] = k_rho
            self.data[band]['kx'] = k_rho*np.cos(self.data[band]['k_phi'])
            self.data[band]['ky'] = k_rho*np.sin(self.data[band]['k_phi'])
            self.data[band]['kz'] = np.array(self.data[band]['k_abs'])*np.cos(self.data[band]['k_theta'])

    def interpolate(self, ndim, resolution=0.01):
        """ work directly in k space with k values (not fraction of brillouin zone) """
        if ndim == 3:
            i_max = np.max(self.data[band]['k_rho'])
            j_max = np.max(self.data[band]['kz'])
            ki_data = self.data[band]['k_rho']
            kj_data = self.data[band]['kz']

        elif ndim = 2:
            i_max = np.max(self.data[band]['kx'])
            j_max = np.max(self.data[band]['ky'])
            ki_data = self.data[band]['kx']
            kj_data = self.data[band]['ky']
        
        noelements_i = int(1/resolution)*i_max/max((i_max, j_max))
        noelements_j = int(1/resolution)*j_max/max((i_max, j_max))
        ki = np.linspace(-0.5,0.5,noelements)
        kj = np.linspace(-0.5,0.5,noelements)