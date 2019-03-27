import csv
import math
import numpy as np
import json
from scipy import interpolate


class CSVLoader(object):
    def __init__(
                self, datafile, file_suffix='undefined', headers=[],
                sort_by='undefined'):
        """
        Base class to dump and analyse Cerenkov data tables from COMSOL.
        Aim is to be adaptable to different datasets.

        Params:
        datafile (str): path of raw csv data
        file_suffix (list<str/int>): Split mode: list of data file suffixes
            for when there is data separated into different files for
            different parameter values. The datafile supplied is the prefix
            i.e. foo_ for foo_100 *without* the .csv extension.
            Will be automatically read and sorted in the data dictionary.
            Single mode: when file_suffix is not supplied or paramter is a
            single string. This only processes data for one param value.
        headers (list<str>): list of paramters featured in data table in order
            of appearance (left to right). Use 'skip' to ignore a column.
            Example: ['band', 'k', 'wavelength', 'angle', 'a'].
        sort_by (str): paramter in headers by which the data dictionary is
            sorted by - sort_by determines the root key. That is the
            outermost index e.g. if sort_by = 'a',
            data = { '100nm' : { <data> } '200nm' : { <data> } ... }.

        Attributes:
        format (str): format of the data: 'split' means separate files for
            each file_suffix paramter and is for automatically looping
            over unit cell size, 'single' means only one parameter is being
            considered and 'merged' means data for different paramters is in
            one file. This is automatically determined.
        data (dict[key = paramter: value = list[float] ]): Data dictionary
            sorted by sort_by paramter as the 'root' key.
        path (str): path of datafile, set in read(), used for automatic graph
            titles if no model name supplied.
        col (dict[key = variable name : column index]): dictionary of column
            numbers for each variable in csv file - for dealing with
            differently formatted data tables.
        prefix (str): Used in split mode, e.g. foo_ in foo_100.csv
        num_bands (dict[key = root]: value = number of bands (int))

        """

        self.prefix = None
        self.data = {}  # dictionary of data, keys are unit cell size
        self.path = None
        self.sort_by = sort_by
        self.col = {}
        self.num_bands = {}

        print("Using CSVLoader")

        # i = 0
        for n, header in enumerate(headers):
            if header is 'skip':
                # i += 1
                continue
            self.col[header] = n
            # self.raw_col[header] = n - i
        print(self.col)
        print("sorting by", self.sort_by)

        if type(file_suffix) is list and file_suffix[0] is not 'undefined':
            self.format = 'split'
            self.prefix = datafile
            self.file_suffix = file_suffix
            print("List of unit cell sizes has been supplied. These will used"
                  "in find_angle automatically unless loop=False.")
            print("NOTE: for the entire dataset (wavelength vs angle), you "
                  "must choose desired value of 'file_suffix' to read by the "
                  "method read(<path>). Otherwise, the first value in the "
                  "list will be used.")
            datafile += str(file_suffix[0]) + ".csv"
        elif type(file_suffix) is str or file_suffix[0] is 'undefined':
            self.file_suffix = [file_suffix]
            self.format = 'single'
        try:
            self.read(datafile)  # use first value in list by default
            if len(self.file_suffix) > 0:
                self.format = 'merged'
        except FileNotFoundError as f:
            raise f
        except OSError:
            raise OSError(
                "Failed to parse filename, ensure format: "
                "'..\\dir\\subdir\\file[.csv]' or '../dir/subdir/file[.csv]'")
        except:
            raise IOError(
                    "Failed to load raw data into dictionary, check the "
                    "format. The variables stored in the dictionary can be "
                    "changed using headers = ['col1', 'col2' ...] where the "
                    "elements are table headings.")

    def _get_raw(self, path):
        with open(path) as file:
            print("Reading file", path)
            reader = csv.reader(file)
            for n_rows, row in enumerate(reader):
                if '%' in ''.join(row):  # skip headings etc.
                    print("ommitting", row)
                    continue
                # if self.format is 'merged' or self.format is 'single':
                try:
                    # root key i.e. self.data[root]
                    root = '%.3g' % float(row[self.col[self.sort_by]])
                except KeyError:
                    raise KeyError("Chosen sort_by key does not exist in list"
                                   "of headers for this data set")
                except IndexError:
                    print("Using supplied value for root key from"
                          "file_suffix' since it is not found in file, or"
                          "table headings are formatted incorrectly")
                    root = self.file_suffix[0]

                # for repeated calls of _get_raw with different root keys
                # e.g. for a, '1.00E-07', '1.50E-07'
                if root not in self.data:
                    self.data[root] = {}
                    self.data[root]['raw'] = []
                self.data[root]['raw'].append(
                    [float(complex(col.replace('i', 'j')).real)
                    for col in row]
                    )  # short CSV, just append
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
            num_bands = int(max([line[self.col['band']] 
                    for line in self.data[root]['raw']]))
            self.num_bands[root] = num_bands
            for key in self.col:
                if key is self.sort_by:
                   continue
                self.data[root][key] = []
                b = 0
                reshape = False
                while b < num_bands:
                    if key is self.sort_by:
                        continue
                    data_dump = [line[self.col[key]]
                            for line in self.data[root]['raw']
                            if int(line[self.col['band']]) == b+1]
                    
                    if self.sort_by is 'band': 
                        b = int(root) - 1 # dont keep appending data
                        self.data[root][key] = data_dump
                    else:
                        self.data[root][key].append(data_dump)
                    b += 1
                # avoid storing param key twice
                # e.g. avoid data['1.00E-07']['1.00E-07']

                # split data into bands (sub lists)
                # print("num", num_bands)
                # self.data[root][key] = [None]*num_bands
                
            self.data[root].pop('raw')

    def sort_data(self, key, direction=1):
        """sort ascending according to key, e.g. key = 'wavelength'"""
        for root in self.data:
            for b in range(self.num_bands[root]):
                data = self.data[root]
                # print("Sorting", key)
                sort_index = np.argsort(data[key][b])
                if direction == -1:
                    sort_index = sort_index[:-1]
                else:  continue

                for param in self.data[root]:
                    # check same data length
                    if len(self.data[root][param][b]) == \
                        self.data[root][key][b]:
                            self.data[root][param][b] = \
                            [data[param][b][ind] for ind in sort_index]

    def save_data(self, name):
        with open(name, 'w') as f:
            json.dump(self.data, f)

#=============================================================================#

class Bzone2D(CSVLoader):
    """Gets all values of kx,ky,(kz) and f from dispersion data and creates
    full Brillouin zone to make a 2D dispersion. Uses interpolation. 
    Implementation for 2D model/crystal exists, but probably has limited 
    use-case"""
    
    def __init__(self, datafile, symmetry, 
                headers=['skip', 'band', 'skip' 'frequency',\
                'kx', 'ky', 'kz', 'n', 'skip'], sort_by = 'band', ndim=3):
        
        print("Using Bzone2D")

        np.set_printoptions(precision=3) # will this affect data saved to text?
        self.ndim = ndim
        self.symmetry = symmetry
        self.reflected = False
        self.status = {'reflected': False, 'interpolated':False, \
            'intersected': False}

        for header in headers:
            if header not in \
            ['band', 'frequency', 'kx', 'ky', 'kz', 'n', 'skip']:
                raise ValueError("Invalid header supplied, must be one of"
                    "['band', 'frequency', 'kx', 'ky', 'kz', 'n']")
        return super(Bzone2D, self).__init__\
            (datafile, headers=headers, sort_by = sort_by)

    def _convert_to_polar(self, inverse=False):
        for band in self.data:
            kx = np.array(self.data[band]['kx'])
            ky = np.array(self.data[band]['ky'])
            kz = np.zeros_like(ky)
            k_theta = np.zeros_like(ky)  # in plane angle
            k_rho = np.sqrt(kx*kx + ky*ky)  # in plane component

            if self.ndim == 2:
                k_abs = k_rho
                for i in range(len(kz)):
                    if k_abs < 1e-10:
                        k_theta[i] = 0.0
                    elif kx[i] < 1e-10 and k_abs[i] > 1e-15:
                        k_theta[i] = np.pi/2.
                    elif kx[i] > 1e-10:
                        k_theta[i] = np.arctan(ky[i]/kx[i])
                k_phi = np.zeros_like(ky)

            if self.ndim == 3:
                kz = np.array(self.data[band]['kz'])
                # k_theta = np.arctan((kx*kx + ky*ky)**0.5/kz)
                k_abs = np.sqrt(kx*kx + ky*ky + kz*kz)
                k_phi = np.zeros_like(kz)
                
                for i in range(len(kx)): # check for kx = 0 case
                    if kx[i] < 1e-10 and k_rho[i] > 1e-10: 
                        print("angle phi is 90 degrees")
                        k_phi[i] = np.pi/2.
                    elif kx[i] < 1e-10 and k_rho[i] < 1e-10:
                        # value doesnt matter, k_rho*anything = 0
                        k_phi[i] = 0.0  
                    # angle of plane our data lies on to kx axis in 3D space
                    elif kx[i] > 1e-15:
                        k_phi[i] = np.arctan(ky[i]/kx[i])
                    
                    if k_abs[i] < 1e-10:
                        k_theta[i] = 0.0
                    elif kz[i] < 1e-10 and k_abs[i] > 1e-10:
                        k_theta[i] = np.pi/2.
                    elif kz[i] > 1e-10:
                        k_theta[i] = \
                            np.arctan((kx[i]*kx[i] + ky[i]*ky[i])**0.5/kz[i])
                # standard deviation ignoring angles for k_rho<1e-10 since
                # these were set to 0.0 and could be wrong
                if np.std(k_phi[k_rho<1e-10]) > 1e-10*np.mean(k_phi):
                    raise ValueError(
                            "Data must be in a form where the BZ "
                            "plane is defined by xz or yz (or (x,y)z with "
                            "y/x fixed - angle between kx and ky must be "
                            "constant")

            # print(type(k_abs), type(k_phi), type(k_theta), type(k_rho))
           
            self.data[band]['k_abs'] = k_abs
            self.data[band]['k_theta'] = k_theta
            self.data[band]['k_rho'] = k_rho
            self.data[band]['k_phi'] = k_phi
            lengths = [len(k_abs), len(k_theta), len(k_rho), len(k_phi)]
            if lengths.count(lengths[0]) != len(lengths):
                raise IndexError("Data arrays do not have"
                                 "same lengths!")

    def reflect(self, symmetry=4):
        """
        Use symmetry of irreducible Brillouin zone to reproduce full Brillouin
        zone. For 3D model, only valid for kx=0 or ky = 0 planes.
    
        Params:
        symmetry (int): Degree of symmetry (of Dihedral group D_n, n=symmetry)
        start_angle (int) Angle (DEGREES) in BZ that data starts at in
            degrees, assumed to be minimum angle
        """
        if self.status['reflected']:
            print("Already reflected!")
            return
        # transformations of angle, first number is multiplier, second is addition
        # reflect in theta = 0 
        reflect_th_0 = [-1.0, 0.0] 
        reflect_th_min45 = [-1.0, -90.0*np.pi/180.0] 
        reflect_th_45 = [-1.0, 90.0*np.pi/180.0]
        reflect_th_90 = [-1.0, np.pi] 
        # but sin(180+theta) = sin(-180+theta)

        if symmetry == 4:
            reflections = [reflect_th_0, reflect_th_90]
        elif symmetry == 8:
            reflections = [reflect_th_0, reflect_th_min45, reflect_th_45]
        elif symmetry == 6 or symmetry == 12:
            raise NotImplementedError(
                "Symmetry of order multiple of 6 not yet implemented"
                "e.g. for the hexagonal unit cell/reciprocal lattice"
                )
        else:
            raise ValueError("Please pick symmetry order 4 or 8")

        self._convert_to_polar()

        for band in self.data:
            start_angle = np.min(self.data[band]['k_theta']) 
            if abs(start_angle) > 1e-10:
                raise ValueError("Please only use data that starts"
                    "from 0 degrees")
            #print(band)
            for params in reflections:
                new_angles = params[0]*np.array(self.data[band]['k_theta']) \
                    + params[1]
                #plane_angle = np.average(self.data[band]['plane_angle'])
                # tile copies array and appends
                self.data[band]['k_abs'] = \
                    np.tile(self.data[band]['k_abs'],2) 
                self.data[band]['k_theta'] = \
                    np.concatenate((self.data[band]['k_theta'], new_angles))
                self.data[band]['frequency'] = \
                    np.tile(self.data[band]['frequency'], 2)
                self.data[band]['k_rho'] = \
                    np.tile(self.data[band]['k_rho'], 2)
                self.data[band]['k_phi'] = \
                    np.tile(self.data[band]['k_phi'], 2)

            # now use full k_abs and k_theta data to derive kx, ky kz
            if self.ndim == 3:
                k_rho = np.array(self.data[band]['k_abs'])*\
                    np.sin(self.data[band]['k_theta'])
                self.data[band]['k_rho'] = k_rho
                self.data[band]['kx'] = k_rho*np.cos(self.data[band]['k_phi'])
                self.data[band]['ky'] = k_rho*np.sin(self.data[band]['k_phi'])
                self.data[band]['kz'] = np.array(self.data[band]['k_abs'])*\
                    np.cos(self.data[band]['k_theta'])
            elif self.ndim == 2:
                self.data[band]['kx'] = self.band[band]['k_abs']*\
                    np.cos(self.data[band]['k_theta'])
                self.data[band]['ky'] = self.band[band]['k_abs']*\
                    np.sin(self.data[band]['k_theta'])
        self.status['reflected'] = True

    def interpolate(self, resolution=100, method='cubic'):
        """ work directly in k space with k values (not fraction of brillouin
        zone) """
        if not self.status['reflected']:
            self.reflect(self.symmetry)

        for band in self.data:
            if self.ndim == 3:
                ki_data = self.data[band]['k_rho']
                kj_data = self.data[band]['kz']

            elif self.ndim == 2:
                ki_data = self.data[band]['kx']
                kj_data = self.data[band]['ky']

            f_data = self.data[band]['frequency']

            print("Hard zeroes in data?", 0 in f_data)
            print("NaNs in data?", math.nan in f_data)
            ki_max = np.max(ki_data)
            ki_min = np.min(ki_data)
            kj_max = np.max(kj_data)
            kj_min = np.min(kj_data)
            
            noelements_i = int(resolution)#*ki_max/max((ki_max, kj_max))
            noelements_j = int(resolution)#*kj_max/max((ki_max, kj_max))
            ki = np.linspace(ki_min, ki_max, noelements_i)
            kj = np.linspace(kj_min, kj_max, noelements_j)
            mi, mj = np.meshgrid(ki,kj)
            mj = np.copy(mj[::-1]) # reverse so that lowest kj is at bottom
            mf = np.zeros_like(mi) # 

            # outside of convex hull the values are NaN, so replace these with
            # nearest neighbour
            interpolate_hull = interpolate.griddata((ki_data,\
                kj_data), f_data, (mi,mj), method=method)
            fill_matrix = interpolate.griddata((ki_data,\
                kj_data), f_data, (mi,mj), method='nearest')
                        
            interpolate_hull[np.isnan(interpolate_hull)] = 0
            mask = np.ones_like(interpolate_hull) * (interpolate_hull==0)
            # invert mask with mask==0
            mf = mask*fill_matrix + interpolate_hull*(mask==0) 
            mf[np.isnan(mf)] = 0
            #print(mf)
            if np.nan in mf:
                raise Warning("NaNs in matrix")

            phi = np.mean(self.data[band]['k_phi'])
            self.data[band]['mx'] = \
                mi*np.cos(phi)
            self.data[band]['my'] = \
                mi*np.sin(phi)
            self.data[band]['mz'] = mj
            self.data[band]['mf'] = mf
            self.data[band]['mi'] = mi
            self.data[band]['mj'] = mj
        self.status['interpolated'] = True

    def addzero(self, band):
        """Add point f(k=0) = 0 to help interpolation"""
        self.data[band]['frequency'].append(0.0)
        self.data[band]['kx'].append(0.0)
        self.data[band]['ky'].append(0.0)
        self.data[band]['kz'].append(0.0)

    def _removerawnans(self):
        """Attempt to fix interpolation with missing data by
        removing data points with NaNs"""
        for band in self.data:
            for i, f in enumerate(self.data[band]['frequency']):
                if math.isnan(f):
                    for param in self.data[band]:
                        if len(self.data[band][param]) == \
                            len(self.data[band]['frequency']):
                                np.delete(self.data[band][param], i)