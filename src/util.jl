export cachefn, cache_dir
using BSON: @save, @load

cache_dir_ = Ref("")
set_cache_dir!(str) = cache_dir_[] = str
cache_dir() = cache_dir_[]

function cachefn(prefix,fn,args...;__oncache__=() -> nothing,kwds...)
    if cache_dir_[] == ""
        @warn "Using default cache directory `$(abspath(cache_dir_))`;"*
            " use EEGCoding.set_cache_dir! to change where results are cached."
    end

    file = joinpath(cache_dir_[],prefix * ".bson")
    if isfile(file)
        __oncache__()
        @load file contents
        contents
    else
        contents = fn(args...;kwds...)
        @save file contents
        contents
    end
end