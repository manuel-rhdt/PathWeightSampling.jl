using HDF5

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
    attrs(group)[name] = repr(value)
end

function write_value_hdf5!(group, name::String, value::Union{AbstractDict, AbstractDataFrame})
    # if we can't write the value directly as a dataset we fall back
    # to creating a new group
    newgroup = g_create(group, name)
    write_hdf5!(newgroup, value)
end

# non array values are written as attributes
function write_value_hdf5!(group, name::String, value::Union{String,Number})
    attrs(group)[name] = value
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