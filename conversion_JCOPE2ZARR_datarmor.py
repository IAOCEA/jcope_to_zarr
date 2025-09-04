#!/usr/bin/env python
# coding: utf-8

# !pip install "zarr<3" -U

# In[1]:


from functoolz_bin import read_first_n_times, read_basic_to_xarray, list_of_time, units_from_base_time,set_time_encoding_hours 
from pathlib import Path
import numpy as np
import xarray as xr


# ## Initialization
# 
# 
# ### Convertion sbasic.dat into xarray

# In[2]:


#Parameters 
path_to_sbasic="/mnt/c/Users/ecap/Documents/JCOPE/sbasic/sbasic.dat"
path_to_sbasic="/Users/todaka/data/jamstec/2103_ctl/sbasic.dat"
jcope_ctr_dir="/Users/todaka/data/jamstec/2103_ctl/"
jcope_bin_dir="/Users/todaka/data/jamstec/jcope2/"

out_path="/Users/todaka/data/jamstec/jcope2.zarr"
out_path="/Users/todaka/data/jamstec/jcope2.zarr"
month="2102"
initial_time="202102251600"
final_time="202102251700"

month="2103"
initial_time="202102220000"
final_time="202102282300"
initial_time="202103010000"
final_time="202103312300"
path_to_sbasic="/scale/project/taos-s/public/jcope/jcope_bin/2103_ctl/sbasic.dat"
jcope_ctr_dir="/scale/project/taos-s/public/jcope/jcope_bin/2103_ctl/"
jcope_bin_dir="/scale/project/taos-s/public/jcope/jcope_bin/"+month
out_path="/scale/project/taos-s/public/jcope/jcope_bin/zarr/tr_5M_20"+month+".zarr"
times = list_of_time(initial_time, final_time)

print(times)
#times=times[1:2]
#times

chunk={"time": 1, "latitude": "auto", "longitude": "5M","z":-1}

im: int = 902
jm: int = 650 
km: int = 47
print(im)

ds_sbasic, fildsc, ichflg = read_basic_to_xarray(path_to_sbasic,im=im,jm=jm,km=km,)

#ds_sbasic.chunk({"longitude":902/2,"latitude":650/2,"zlev":"10M"}).to_zarr('jcope.zarr',mode='w')
ds_sbasic


# Now we have converted the sbasic into xarray.dataset we have Z , ZZ (layers thickness), DZ 
# 
# zlev(kzd) →  layer centers
# 
# z_z(kzd+1) → depths of the interfaces.
# 
# dz_z(kzd) → vertical thickness of layers.

# ## Transform  first time step to zarr ( initialise the Zarr file)

# In[6]:


tt_ctl = Path(jcope_ctr_dir) / "TT.ctl"
egt_ctl = Path(jcope_ctr_dir) / "EGT.ctl"
egtdir = Path(jcope_bin_dir) 
ttdir = Path(jcope_bin_dir) 



n_time=1
#ttdir
#ds_egt = read_first_n_times(egtdir, egt_ctl, "EGT_*.bin", n=n_time, no_z=True)
# -------------------

# Lire les n premiers pas de temps 

ds_egt = read_first_n_times(egtdir, egt_ctl, "EGT_"+initial_time+".bin", n=n_time, no_z=True)

ds_tt  = read_first_n_times(ttdir, tt_ctl,  "TT_"+initial_time+".bin", n=n_time)
ds_combined_ssh_tt = xr.merge([ds_egt, ds_tt])
ds_combined_ssh_tt = ds_combined_ssh_tt.rename({"x": "longitude", "y": "latitude", "sigma": "z"})
ds_combined_ssh_tt=ds_combined_ssh_tt.transpose("time", "latitude", "longitude", "z")

# Replace coords directly
ds_sbasic = ds_sbasic.assign_coords(
    longitude = ds_combined_ssh_tt.longitude,
    latitude  = ds_combined_ssh_tt.latitude
)

ds=xr.merge([ds_combined_ssh_tt, ds_sbasic]).chunk(chunk)
ds=ds.rename_vars({"Z":"_Z"})
# ds0 は最初の 1 ステップ（あなたの ds0）を想定
base_time = ds['time'].values[0]                   # ここが「イニシャルタイム」
units = units_from_base_time(base_time)            # 例: "hours since 2021-02-22 00:00:00"
ds = set_time_encoding_hours(ds, units)
time_encoding={'time': {'units': units}}

ds.to_zarr(out_path, mode='w',zarr_version=2 ,    consolidated=True,
    encoding=time_encoding
)
ds


# ## Make for loop of time , and load data of that time, append it to the final zarr file  

# In[ ]:






# In[9]:


for time in times:
    print('appeding time',time)
    ds_egt = read_first_n_times(egtdir, egt_ctl, "EGT_"+time+".bin", n=n_time, no_z=True)
    
    ds_tt  = read_first_n_times(ttdir, tt_ctl,  "TT_"+time+".bin", n=n_time)
    ds = xr.merge([ds_egt, ds_tt])
    ds = ds.rename({"x": "longitude", "y": "latitude", "sigma": "z"})
    ds=ds.transpose("time", "latitude", "longitude", "z")
    ds=ds.chunk(chunk)
    ds = set_time_encoding_hours(ds, units)

    ds.to_zarr(out_path,  mode='a', append_dim='time',zarr_version=2 
               ,  consolidated=True,
              )


# In[10]:


xr.open_zarr(out_path).time


# In[ ]:




