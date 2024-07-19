
# Output Saving

This package provides functions for saving data to disk in convenient formats, supporting both JSON and HDF5. The output of `mutual_information` can be efficiently saved to disk using these functions.

## JSON Output

The `write_json` function saves data from a dictionary to a specified directory, creating individual JSON files for each entry. This format is human-readable and easily shareable, ideal for cases where data needs to be inspected or transferred without specialized tools.

### Usage
- **Function**: `write_json`
- **Input**: A directory path and a dictionary.
- **Output**: JSON files for each key-value pair in the dictionary.

## HDF5 Output

The `write_hdf5!` function allows for efficient storage of complex data structures into an HDF5 file. This format is highly suitable for large datasets due to its support for hierarchical data organization and fast I/O operations.

### Usage
- **Function**: `write_hdf5!`
- **Input**: An HDF5 group and either a dictionary or a DataFrame.
- **Output**: Data is stored in the HDF5 format, with nested structures represented as groups.

## API

```@docs
PathWeightSampling.write_json
PathWeightSampling.write_hdf5!
```