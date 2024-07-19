using HDF5

"""
Writes data from a dictionary or DataFrame to an HDF5 group.

# Arguments
- `group`: The HDF5 group where the data will be stored.
- `dict::AbstractDict`: A dictionary containing data to write, with keys as names and values as data.
- `df::AbstractDataFrame`: A DataFrame containing data to write, with column names as keys.

# Description
The function `write_hdf5!` serializes data from either a dictionary or a DataFrame into the specified HDF5 group. 
For dictionaries, each key-value pair is processed, writing the value to the HDF5 group under the corresponding key name. 
For DataFrames, each column is written using the column name as the key.

# Behavior
- For simple types like strings and numbers, values are stored as attributes.
- For numerical arrays, values are stored directly in the group.
- For more complex types (dictionaries or DataFrames), new groups are created recursively.

# Example
```julia
using HDF5, DataFrames

data = Dict("numbers" => [1, 2, 3], "info" => Dict("a" => 10, "b" => 20))
df = DataFrame(a = 1:3, b = 4:6)

h5open("data.h5", "w") do file
    write_hdf5!(file, data)
    write_hdf5!(file, df)
end
```
"""
function write_hdf5!(group, dict::AbstractDict)
    for (name, value) in dict
        name = String(name)
        write_value_hdf5!(group, name, value)
    end
end

function write_hdf5!(group, df::AbstractDataFrame)
    for (name, value) in zip(names(df), eachcol(df))
        write_value_hdf5!(group, String(name), value)
    end
end

function write_value_hdf5!(group, name::String, value)
    # for unknown datatypes write a string representation of the value as an attribute
    attributes(group)[name] = repr(value)
end

function write_value_hdf5!(group, name::String, value::Union{AbstractDict, AbstractDataFrame})
    # if we can't write the value directly as a dataset we fall back
    # to creating a new group
    newgroup = create_group(group, name)
    write_hdf5!(newgroup, value)
end

# non array values are written as attributes
function write_value_hdf5!(group, name::String, value::Union{String,Number})
    attributes(group)[name] = value
end

function write_value_hdf5!(group, name::String, value::AbstractArray{<:Number})
    group[name] = value
end

function write_value_hdf5!(group, name::String, value::AbstractVector{<:Array{T,N}}) where {T,N}
    outer_len = length(value)
    if outer_len < 1
        group[name] = zeros(T, 0)
        return
    end
    inner_size = size(value[1])
    dset = create_dataset(group, name, datatype(T), dataspace(inner_size..., outer_len), chunk=(inner_size..., 1))
    for (i, subarray) in enumerate(value)
        dset[axes(subarray)..., i] = subarray
    end
end