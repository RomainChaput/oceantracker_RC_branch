from netCDF4 import Dataset as DatasetNCDF
import  xarray as xr
from oceantracker.util.ncdf_util import NetCDFhandler
from time import  perf_counter
from os import  path, listdir
from glob import glob
import numpy as np

mask=r'G:\Hindcasts_large\2020_MalbroughSounds_10year_benPhD\\2008\schism_marl2008010*.nc'
mask=r'G:\Hindcasts_large\2020_MalbroughSounds_10year_benPhD\\2008\schism_marl2008010*.nc'

files= glob(mask,recursive=True)
print('numfiles=',len(files))

t0= perf_counter()
nc= xr.open_mfdataset(mask,coords =['time'])
nc['time'] = nc['time'].astype('datetime64[s]')
t=nc['time']

#nc.to_zarr(path.join(r'F:\temp\zarr_test',path.basename(fn).rsplit('.',)[0]))
nc.close()
print('open_mfdataset', perf_counter()-t0)
print(np.diff(t)/1e9)

