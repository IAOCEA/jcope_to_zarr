#!/usr/bin/env python
# coding: utf-8
!pip install "zarr<3" -U
# In[1]:


from functoolz_bin import read_first_n_times, read_basic_to_xarray, list_of_time
from pathlib import Path
import numpy as np
import xarray as xr


# ## Initialization
# 
# 
# ### Convertion sbasic.dat into xarray

# In[2]:


im: int = 902
jm: int = 650 
km: int = 47
print(im)
path_to_sbasic="/mnt/c/Users/ecap/Documents/JCOPE/sbasic/sbasic.dat"
path_to_sbasic="/Users/todaka/data/jamstec/2103_ctl/sbasic.dat"
path_to_sbasic="/home/datawork-taos-s/public/jcope/jcope_bin/2103_ctl/sbasic.dat"
ds_sbasic, fildsc, ichflg = read_basic_to_xarray(path_to_sbasic,im=im,jm=jm,km=km,)

#ds_sbasic.chunk({"longitude":902/2,"latitude":650/2,"zlev":"10M"}).to_zarr('jcope.zarr',mode='w')
ds_sbasic

ds_sbasic.Z.isel(z=1).plot()
# Now we have converted the sbasic into xarray.dataset we have Z , ZZ (layers thickness), DZ 
# 
# zlev(kzd) →  layer centers
# 
# z_z(kzd+1) → depths of the interfaces.
# 
# dz_z(kzd) → vertical thickness of layers.

# ## Download from jamstec if needed
# 
from download_bin import fetch_hourly_and_decompress


output_base_dir = Path("/mnt/c/Users/ecap/Documents/JCOPE/4days_data_jcope_202102/")
rr
# URL du dossier distant 
base_url = "https://www.jamstec.go.jp/jcope/data/odaka20240424-2102"

# Date de départ 
start_time = "202102220000"
n_time = 2
# overwrite=False évite de retélécharger / regunzipper si déjà présent
overwrite_downloads = False

# strict=True lèvera une erreur si on n'a pas trouvé exactement n_time fichiers
strict_mode = True
# ---------------------------------------------------------

egtdir_new = output_base_dir / "EGT"
ttdir_new  = output_base_dir / "TT"

fetch_hourly_and_decompress("EGT", base_url, start_time, n_time, egtdir_new,
                            overwrite=overwrite_downloads, strict=strict_mode)
fetch_hourly_and_decompress("TT",  base_url, start_time, n_time, ttdir_new,
                            overwrite=overwrite_downloads, strict=strict_mode)

egtdir = egtdir_new
ttdir  = ttdir_new
# ## Transform  first time step to zarr ( initialise the Zarr file)

# In[3]:


#egt_ctl = egtdir.parent / "CTL/EGT_patch_zdef1.ctl"
#tt_ctl = ttdir.parent / "CTL/TT.ctl"
#out_path = ttdir / "EGT_TT.zarr"

jcope_ctr_dir="/Users/todaka/data/jamstec/2103_ctl/"
jcope_ctr_dir="/home/datawork-taos-s/public/jcope/jcope_bin/2103_ctl/"


jcope_bin_dir="/Users/todaka/data/jamstec/jcope2/"
jcope_bin_dir="/home/datawork-taos-s/public/jcope/jcope_bin/2102"


out_path="/Users/todaka/data/jamstec/jcope2.zarr"
out_path="/Users/todaka/data/jamstec/jcope2.zarr"
out_path="/home/datawork-taos-s/public/jcope/jcope_bin/zarr/tr.zarr"

tt_ctl = Path(jcope_ctr_dir) / "TT.ctl"
egt_ctl = Path(jcope_ctr_dir) / "EGT.ctl"
egtdir = Path(jcope_bin_dir) 
ttdir = Path(jcope_bin_dir) 
chunk={"time": 1, "latitude": "auto", "longitude": "5M","z":-1}


initial_time="202102251600"
initial_time="202102220000"


n_time=1
#ttdir
#ds_egt = read_first_n_times(egtdir, egt_ctl, "EGT_*.bin", n=n_time, no_z=True)


# In[4]:


# -------------------

# Lire les n premiers pas de temps 

ds_egt = read_first_n_times(egtdir, egt_ctl, "EGT_"+initial_time+".bin", n=n_time, no_z=True)

ds_tt  = read_first_n_times(ttdir, tt_ctl,  "TT_"+initial_time+".bin", n=n_time)


# In[5]:


ds_combined_ssh_tt = xr.merge([ds_egt, ds_tt])
ds_combined_ssh_tt = ds_combined_ssh_tt.rename({"x": "longitude", "y": "latitude", "sigma": "z"})
ds_combined_ssh_tt=ds_combined_ssh_tt.transpose("time", "latitude", "longitude", "z")


# In[6]:


# Replace coords directly
ds_sbasic = ds_sbasic.assign_coords(
    longitude = ds_combined_ssh_tt.longitude,
    latitude  = ds_combined_ssh_tt.latitude
)


# In[7]:


ds=xr.merge([ds_combined_ssh_tt, ds_sbasic]).chunk(chunk)
ds=ds.rename_vars({"Z":"_Z"})

ds.to_zarr(out_path, mode='w',zarr_version=2 )
ds


# In[8]:


xr.open_zarr(out_path)


# ## Make for loop of time , and load data of that time, append it to the final zarr file  

# In[9]:


# Example usage
times = list_of_time("202102220000", "202102282300")
#print(times)
#times=times[1:2]
times


# In[10]:


for time in times:
#time='202102251700'
    print('appeding time',time)
    ds_egt = read_first_n_times(egtdir, egt_ctl, "EGT_"+time+".bin", n=n_time, no_z=True)
    
    ds_tt  = read_first_n_times(ttdir, tt_ctl,  "TT_"+time+".bin", n=n_time)
    ds_combined_ssh_tt = xr.merge([ds_egt, ds_tt])
    ds_combined_ssh_tt = ds_combined_ssh_tt.rename({"x": "longitude", "y": "latitude", "sigma": "z"})
    ds_combined_ssh_tt=ds_combined_ssh_tt.transpose("time", "latitude", "longitude", "z")
    ds_combined_ssh_tt=ds_combined_ssh_tt.chunk(chunk)
    ds_combined_ssh_tt.to_zarr(out_path,  mode='a', append_dim='time',zarr_version=2 )


# In[11]:


xr.open_zarr(out_path)


# In[ ]:




