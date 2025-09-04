# jcope_to_zarr

 micromamba create -n jcope2zarr python=3.11 xarray dask zarr netcdf4 h5netcdf pandas numpy ipython jupytext matplotlib

qsub -I -q mpi_1 -l walltime='24:00:00'

## How to run

python ./jcope_bin_to_netcdf.py --ctl ./TT.ctl --bin TT_202102251600.bin --out TT_202102251600.nc
