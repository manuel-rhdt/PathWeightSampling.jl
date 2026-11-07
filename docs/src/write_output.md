# Saving Output

`write_json` and `write_hdf5!` save results from `mutual_information` and other nested data.

## JSON Output

`write_json(path, data)` writes each entry of a dictionary to a separate JSON file in `path`, creating the directory if needed. JSON is useful when you want to inspect results in a text editor or read them without an HDF5 library. DataFrames are written as gzip-compressed, line-delimited JSON.

## HDF5 Output

`write_hdf5!(group, data)` writes a dictionary or DataFrame to an HDF5 group. Nested dictionaries and DataFrames become groups; numerical arrays become datasets. HDF5 is usually more practical for large numerical datasets and preserves their hierarchical structure.

## API

```@docs
PathWeightSampling.write_json
PathWeightSampling.write_hdf5!
```
