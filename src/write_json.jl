
import JSON
import GZip
import DataFrames: AbstractDataFrame

"""
Writes the contents of a dictionary to separate JSON files in a specified directory.

# Arguments
- `path::AbstractString`: The directory path where JSON files will be created.
- `val::AbstractDict`: A dictionary with keys as filenames and values as data to be serialized.

# Description
Creates the directory at `path` if it does not exist. Iterates over each key-value pair in `val`, 
writes each value to a separate JSON file named after the key in the specified directory.
"""
function write_json(path::AbstractString, val::AbstractDict)
    mkpath(path)
    for (k, v) in val
        new_path = joinpath(path, string(k))
        write_value_json(new_path, v)
    end
end

write_json(path::AbstractString, val::Union{Tuple, NamedTuple}) = write_json(path, pairs(val))

"""
Writes a value to a JSON file.

# Arguments
- `path::AbstractString`: The file path (without extension) where the JSON data will be written.
- `val::Any`: The value to be serialized into JSON format.

# Description
Serializes the given `val` into JSON format and writes it to a file at `path` with the `.json` extension.
"""
function write_value_json(path::AbstractString, val::Any)
    file = open(path * ".json", "w")
    JSON.print(file, val)
    close(file)
end

"""
    function write_value_json(path::AbstractString, df::AbstractDataFrame)
    
Writes the contents of a `DataFrame` to a compressed JSON file.

# Arguments
- `path::AbstractString`: The file path (without extension) where the JSON data will be written.
- `val::AbstractDataFrame`: The DataFrame containing data to be serialized into JSON format.

# Description
The function iterates over each row of the DataFrame `df`, converts it to JSON format, 
and writes it to a file at `path` with the `.json.gz` extension. Each row is written as 
a separate line in the JSON file. The file is compressed using GZip.
"""
function write_value_json(path::AbstractString, df::AbstractDataFrame)
    file = GZip.open(path * ".json.gz", "w")
    for row in eachrow(df)
        JSON.print(file, row)
        write(file, "\n")
    end
    close(file)
end