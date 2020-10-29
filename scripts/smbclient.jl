using Logging

const date_re = r"(\w{3}\s\w{3}\s{1,2}\d{1,2}\s\d{2}:\d{2}:\d{2}\s\d{4})"

const file_re = r"\s{2}"*         # file lines start with 2 spaces
    r"(.*?)\s+"*            # capture filename non-greedy, eating remaining spaces
    r"([ADHSR]*)"*          # capture file mode
    r"\s+"*                 # after the mode you can have any number of spaces
    r"(\d+)"*               # file size
    r"\s+"*                 # spaces after file size
    date_re*
    r"$"

const path_dir_splitter = r"^(.*?)([/\\]+)([^/\\]*)$"

mutable struct SmbPath
    host::String
    auth_file::String
    path::String
end

function run_smbclient(path::SmbPath, command::String; dir=nothing)
    if dir === nothing
        cmd = `smbclient -A $(path.auth_file) $(path.host) -c $command`
    else 
        cmd = `smbclient -A $(path.auth_file) $(path.host) -D $dir  -c $command`
    end
    @info "smbclient" cmd
    output = readlines(cmd)
end

function Base.Filesystem.readdir(path::SmbPath)
    output = run_smbclient(path, "ls", dir=path.path)
    result = String[]
    for l in output
        res = match(file_re, l)
        if res !== nothing && res[1] != "." && res[1] != ".."
            push!(result, res[1])
        else
            nothing
        end
    end
    result
end

function allinfo(path::SmbPath)
    smb_path = path.path
    output = run_smbclient(path, "allinfo $smb_path")
    infodict = Dict{String, Any}()
    for l in output
        attr_re = r"attributes:\s+([ADHSR]*)\s+\([0-9]+\)"
        res = match(attr_re, l)
        if res !== nothing
            infodict["attributes"] = res[1]
        end
    end
    infodict
end

function Base.Filesystem.isdir(path::SmbPath)
    info = allinfo(path)
    if haskey(info, "attributes")
        'D' in info["attributes"]
    else
        false
    end
end

function Base.Filesystem.mkdir(path::SmbPath, name::String)
    output = run_smbclient(path, "mkdir $name", dir=path.path)
end

function Base.Filesystem.rm(path::SmbPath; recursive::Bool=false)
    if recursive
        for (root, dirs, files) in walkdir(path, topdown=false)
            for dir in dirs
                rm(joinpath(root, dir))
            end
            for file in files
                rm(joinpath(root, file))
            end
        end
    end

    dir, name = splitdir(path)
    if isdir(path)
        command = "rmdir $name"
        if !isempty(readdir(path))
            error("directory is not empty")
        end
    else
        command = "rm $name"
    end
    output = run_smbclient(path, command, dir=dir.path)
end

function Base.Filesystem.mkdir(path::SmbPath)
    dname, basename = Base.Filesystem.splitdir(path)
    mkdir(dname, basename)
end

function Base.Filesystem.cp(src::String, dst::SmbPath)
    dir, name = splitdir(dst)
    run_smbclient(dst, "put $src $name", dir=dir.path)
end
 
function Base.Filesystem.joinpath(root::SmbPath, parts::AbstractString...)
    comps = [root.path, parts...]
    SmbPath(root.host, root.auth_file, join(comps, '\\'))
end

function splitdir_str(a::String, b::String)
    m = match(path_dir_splitter,b)
    m === nothing && return (a,b)
    a = string(a, isempty(m.captures[1]) ? m.captures[2][1] : m.captures[1])
    a, String(m.captures[3])
end

function Base.Filesystem.splitdir(path::SmbPath)
    (a, b) = splitdir_str("", path.path)
    SmbPath(path.host, path.auth_file, a), b
end

function Base.Filesystem.dirname(path::SmbPath)
    dirn = splitdir_str("", path.path)[1]
    SmbPath(path.host, path.auth_file, dirn)
end

function Base.Filesystem.basename(path::SmbPath)
    splitdir("", path.path)[2]
end

function Base.Filesystem.mkpath(path::SmbPath)
    if !isdir(path)
        dname = Base.Filesystem.dirname(path)
        mkpath(dname)
        mkdir(path)
    end
end

Base.Filesystem.islink(path::SmbPath) = false

Base.Filesystem.isabspath(path::SmbPath) = true

host = "//sun.amolf.nl/tenwolde"
sun_home = SmbPath(host, "/home/ipausers/reinhardt/sun_auth.txt", "home-folder\\reinhardt")
