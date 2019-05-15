export cachefn
using BSON: @save, @load

cache_dir = Ref("")
set_cache_dir!(str) = cache_dir[] = str

function cachefn(prefix,fn,args...;oncache=() -> nothing,kwds...)
    if cache_dir[] == ""
        @warn "Using default cache directory `$(abspath(cache_dir))`;"*
            " use EEGCoding.set_cache_dir! to change where results are cached."
    end

    file = joinpath(cache_dir[],prefix * ".bson")
    if isfile(file)
        oncache()
        @load file contents
        contents
    else
        contents = fn(args...;kwds...)
        @save file contents
        contents
    end
end