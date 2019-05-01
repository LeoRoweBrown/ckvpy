# ckvpy
Python analysis package for wave optics simulation data, speficially for periodic structures that exhibit Cerenkov-like behaviour.

# 2D Model Documentation
## `Anaylze2D` Class
For 2D models, the class Analyzez2D() is used to create plots for CSV data that came from COMSOL. It is the main
thing you will be using to analyze data.
The paramters that this class takes (in order) are: 

* `datafile` (str): Path of raw csv data.
* `headers` (list (str)) - optional: List of parameters featured in CSV data table (exported from COMSOL) in order
of appearance (left to right). Use 'skip' to ignore a column.
The default for Analyze2D is `headers = ['a', 'band', 'k', 'frequency', 'wavelength', 'angle', 'kx', 'ky', 'kz']`.
* `sort_by` (str) - optional:
paramter in headers by which the data dictionary is sorted by - `sort_by` determines the 'root key'. That is the
first level of indexing e.g. if sort_by = 'a', and the values of 'a' are = '100nm', '200nm' ... then
`data = { '100nm' : { <data> } '200nm' : { <data> } ... }`.
By default this is 'a', the unit cell size.

## Methods of `Analyze2D`
`plot_cherenkov(self, filename=None, modelname=None, a_i=0, a=None, dump=False, bands=None)`

This plots wavelength against angle outside of the crystal for the root key called 'a', or
the `a_i`th root key (unit cell size by default). Indexing the keys like this only works
when the keys are ordered in a dictionary, i.e. python 3.7+.

Parameters:

* `filename` (str) optional: Filename of exported graph. Just show graph if not specified.
* `modelname` (str) optional: String appended to title to distinguish graphs of
        different models.
* `band` (int) optional: Band for which data is plotted.
* `a_i` (int) optional: Index of root key, i.e. `a_i`th element ordered dictionary keys.
* `a` (str) optional: Alternative to `a_i`, supply key directly.
* `bands` (list) (int/str)) optional: Bands to plot, if None plot all (count from 0)

---

`plot_a(self, filename=None, modelname=None, band='0')`

This plots the average Cherenkov angle and chromatic error (as error bars) against unit cell size, a.
Before running this, run `data.find_angle()` - see below. 
Parameters:

* `filename` (str) optional: Filename of exported graph. Just show graph if not specified.
* `modelname` (str) optional: String appended to title to distinguish graphs of different models.
* `band` (str/int) optional: Band to plot for, default only for first band.

---

```
def compare_sio2(
    self, ratio=None, index="sio2", filename=None, modelname=None,
    n_lim=None, beta=0.999, roots=None, bands=None):
```

Plot graphs comparing the refractive indices derived from the data
and given a Maxwell-Garnett approximation. Default behaviour plots for all root keys (all unit cell sizes) and all bands.
    
Parameters:

* `ratio` (float): Volume ratio between air and SiO2.
* `index` (str): material refractive index in `./index/<index>.txt`
    used to calculate expected index from effective medium.
* `modelname` (str): Used in title of graph to identify model.
* `filename` (str): Used to decide filename (has index of a and 
        band appended, i.e. for a = 1e-7, band = 0 filename + \_a\_1e-7b\_0).
* `n_lim` (list (float)): refractive index n plot range (y axis range).
* `roots` (list (int): List of root keys to plot and save for. If `None` do all.
* `bands` (list (str): List of bands to plot and save for. If `None` do all.

## `dataAnalysis` (and `dataAnalysis3D`) Class
`Analyze2D` (`Analyze3D`) have the member attribute `data` which is an instance
of `dataAnalysis` (`dataAnalysis3D`). This object in turn has the attribute
`data_dict` which is the dictionary mentioned previously that contains all the data.
In the 3D case there is also the dictionary `data_full` which contains the 3D dispersion
relation (projected into a 2D plane such as kx,kz), whereas `data_dict` just has the
intersection of the electron plane and this dispersion, deriving the Cherenkov behaviour.
Although there are other methods and these are documented with docstrings, here are the most
useful/important methods in these classes:

`find_angle(self, wl_range=[250.0e-9, 500.0e-9], filename=None)`

This finds the average Cherenkov angle and chromatic error (Cherenkov angle range) for
a given wavelength range for each root key, i.e. for the 2D case it calculates these values for
each value of a. This is stored in `data_dict[root][band]['cherenkov']`. This method must be
run before running plot_a.

Parameters:

* wl_range list(float): Wavelength range that Cherenkov angle and chromatic error is calculated over.

---

`save(self, filename)`

Saves data_dict as a json file with the name `filename`.

---

`save_cherenkov(self, filename)`

Saves cherenkov data from `find_angle()` as a table txt file with the name `filename`.

`calculate_yield(self, beta=0.999, L=100.e-6, wl_range=[250.e-9, 500.e-9])`

Calculate the number of photons produced by the photonic crystal radiator in a given wavelength range, radiator length,
and electron speed.

Parameters:

* beta (float) - optional: Ratio of particle speed to speed of light
* L (float) - optional: Radiator length
* wl_range (list (float)): Wavelength range that yield is computed over

---

### Analysis3D Only

`calculateCherenkov(self, beta=0.999, direction = [1,0])`

Perform the intersection between the cherenkov electron/particle plane and the dispersion relation (omega, angular frequency
against k-vector) of the photonic crystal:
this gives the Cherenkov behaviour. This isn't required in the 2D version since COMSOL iteratively calculates
the dispersion relation and the electron plane together as a non-linear problem. This means that the 3D version includes data
to solve for any value of beta - just rerun `calculateCherenkov`. This method dumps the results into data_dict so
that it behaves the same as `Analyze2D` once this has been run.

Parameters:
* beta (float) - optional: Ratio of particle speed to speed of light
* direction (list (int)): Direction of particle trajectory: the indices represent rho (|x,y|) and z respectively. 
This defines particle plane: omega = k.v. At the moment only `[1,0]` and `[0,1]` are supported i.e. kz.v, k_rho.v - the particle
must be confined to (x,y) plane or z. z is usually defined as the direction into the plane in COMSOL (out-of-plane) e.g. the direction that a 2D work-plane
is extruded, while rho is the magnitude of the in-plane vector.

## Example for 2D Model
First import the package and create an obect
```import ckvpy
a = ckvpy.Analyze2D(<csv path>)
```

To simply plot wavelength against Cherenkov angle for the 1st root key 
(i.e. first value of a in the dictionary, which is 50nm with the original dataset)
and save it as `filename`, and to add the name of the model, `modelname`, to the title of the graph so that it reads 
"Saturated Cherenkov Angle against Wavelength a = _unit cell size in metres_ m ( _modelname_ )":

```python
a.plot_cherenkov(filename, modelname, a_i=1)
```

To plot Cherenkov angle and chromatic error against unit cell size, first derive the average Chernekov angle 
and chromatic error (Cherenkov angle range) in the wavelength range for the dataset:

```python
a.data.find_angle(wl_range=[300e-9, 500e-9])
a.a_plot(filename, modelname)
```

To compare to Maxwell Garnett and volume-weighted average theory for refractive index inside crystal

```python
a.compare_sio2(ratio_2d=0.1 (optional), modelname (optional), filename= (optional),
               index="sio2" (optional), bands=[None] (optional), a_s=[None] (optional)))
```

# 3D Model Documentation (unfinished)

## `Analyze3D` Class

This class is similar to `Analyze3D`, except there are a few differences: One can plot the full dispersion relation
of the photonic crystal since the method required to find Cherenkov behaviour in the 3D model neccessitates
solving the dispersion relation omega(kx,ky,kz) fully, then taking the intersection of a particle/electron plane to find
Cherenkov modes. Since multiple root keys hasn't been implemented (like in Analyze2D where data for many different values
of 'a' are sorted and stored in the dictionary), methods like `a_plot()` do not exist.

## Methods of `Analyze3D`

`plot_cherenkov(self, mode='surface')`:

Has similar functionality to the 2D counterpart, however filenames and modelnames haven't quite been implemented. Instead
'cherenkov.png' and 'intersection.png' are hard-coded. There are two plots: one which is the regular wavelength against angle
and one which shows the data points projected in 3D which show how the intersection between the particle plane and the photonic
crystal's dispersion give these data points. One should see that they exist in the line which make a constant angle with kz and k_rho 
and hence constant Cherenkov angle. There are the graphs named 'cherenkov.png' and 'intersection.png' respectively.

`plot_3d(self)`

Plots the 3D dispersion relation of the 3D crystal - frequency (not angular) against kz and k_rho. They are saved as `"dispersion"+band+".png"`.

Parameters:

* mode (str): Type of plot - default is `'surface'` which gives the regular interpolated 3D dispersion relationship, whereas `'scatter'` shows the raw
data points projected in 3D.

```
compare_sio2(
    self, ratio=0.106, index="sio2", filename=None, modelname=None,
    n_lim=None, roots=None, bands=None)
```

Same as Analyze2D version.

## Example for 3D Model

First import the package and create an obect
```import ckvpy
a = ckvpy.Analyze2D(<csv path>)
```

To plot the 3D dispersion of the 3D model (actually confined to 2D - rho and z), you woulld run

```python
a.plot_3d(filename, modelname, a_i=1)
```

To plot Cherenkov angle and chromatic error against unit cell size, first derive the average Chernekov angle 
and chromatic error (Cherenkov angle range) in the wavelength range for the dataset:

```python
a.data.find_angle(wl_range=[300e-9, 500e-9])
a.a_plot(filename, modelname)
```

To compare to Maxwell Garnett and volume-weighted average theory for refractive index inside crystal

```python
a.compare_sio2(ratio_2d=0.1 (optional), modelname (optional), filename= (optional),
               index="sio2" (optional), bands=[None] (optional), a_s=[None] (optional)))
```

# Another planned feature
inside analysis.py, since to access data you must get the instance data from the data object which is called
self.data in analyze2D and analyze3D, this means self.data.data is typed. May override `__getitem__` and `__setitem__`
to allow direct indexing of the object such that self.data[key] does self.data.data[key], but this hides behaviour?