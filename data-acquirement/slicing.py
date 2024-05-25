# this is for the dataset given on hugging face : 
# https://huggingface.co/datasets/Zeel/P1/blob/main/all_in_one.zarr.zip

# Download local copy -> unzip -> copy path to .zarr file (line 7)
import xarray as xr
import zarr
ds = xr.open_zarr("path/to/zarr/file/all_in_one.zarr")
# choose start slice and end slice
time_frame = slice("2023-11-01", "2023-11-30")
sliced_ds = ds.sel(Timestamp=time_frame)
sliced_ds = sliced_ds.sel(station=ds.station.state == "Delhi")

# filename specified here
sliced_ds.to_netcdf("delhi_nov.nc")