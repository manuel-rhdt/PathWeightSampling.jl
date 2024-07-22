
import Parquet2
import DataFrames: AbstractDataFrame

"""
Writes the contents of a dictionary to separate Parquet files in a specified directory.

# Arguments
- `path::AbstractString`: The directory path where Parquet files will be created.
- `val::AbstractDict`: A dictionary with keys as filenames and values as data to be serialized.

# Description
Creates the directory at `path` if it does not exist. Iterates over each key-value pair in `val`, 
writes each value to a separate Parquet file named after the key in the specified directory.
"""
function write_parquet(path::AbstractString, val::AbstractDict)
    mkpath(path)
    for (k, v) in val
        new_path = joinpath(path, string(k))
        write_value_parquet(new_path, v)
    end
end

write_parquet(path::AbstractString, val::Union{Tuple, NamedTuple}) = write_parquet(path, pairs(val))

"""
Writes a value to a Parquet file.

# Arguments
- `path::AbstractString`: The file path (without extension) where the JSON data will be written.
- `val::Any`: The value to be serialized into JSON format.

# Description
Serializes the given `val` into JSON format and writes it to a file at `path` with the `.json` extension.
"""
function write_value_parquet(path::AbstractString, val::Any)
    write_value_json(path, val)
end

"""
    function write_value_parquet(path::AbstractString, df::AbstractDataFrame)
    
Writes the contents of a `DataFrame` to a Parquet file.

# Arguments
- `path::AbstractString`: The file path (without extension) where the JSON data will be written.
- `val::AbstractDataFrame`: The DataFrame containing data to be serialized into JSON format.

# Description
The function iterates over each row of the DataFrame `df`, converts it to JSON format, 
and writes it to a file at `path` with the `.json.gz` extension. Each row is written as 
a separate line in the JSON file. The file is compressed using GZip.
"""
function write_value_parquet(path::AbstractString, df::AbstractDataFrame)
    filename = path * ".parquet"
    Parquet2.writefile(filename, df)
end