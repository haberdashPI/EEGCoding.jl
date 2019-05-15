

Base.@kwdef struct EEGData
    label::Vector{String}
    fs::Int
    data::Vector{Matrix{Float64}}
end

SampledSignals.samplerate(x::EEGData) = x.fs
function SampledSignals.samplerate(x::MxArray)
    fs = mat"$x.fsample"
    fs
end

function eegtrial(x::MxArray,i)
    mat"response = $x.trial{$i};"
    response = get_mvariable(:response)
    response
end

function eegtrial(x::EEGData,i)
    x.data[i]
end

################################################################################
# handle selecting various bounds of a signal

# denotes selection of all valid time indices
struct AllIndices end
const all_indices = AllIndices()
Base.getindex(x::AllIndices,i::Int) = x
Base.isempty(x::AllIndices) = false
(x::AllIndices)(row) = all_indices

struct NoIndices end
const no_indices = NoIndices()
Base.isempty(x::NoIndices) = true

toindex(x,min,fs) = clamp.(round.(Int,x.*fs),1,min)

function select_bounds(x::AbstractArray,::AllIndices,min_len,fs,dim)
    if dim == 1
        x[1:min(min_len,end),:]
    elseif dim == 2
        x[:,1:min(min_len,end)]
    end
end

function select_bounds(x::MxArray,::AllIndices,min_len,fs,dim)
    if dim == 1
        mat"x = $x(1:min(end,$min_len),:);"
    elseif dim == 2
        mat"x = $x(:,1:min(end,$min_len));"
    end
    get_mvariable(:x)
end

function select_bounds(x::AbstractArray,(start,stop)::Tuple,min,fs,dim)
    start,stop = toindex.((start,stop),min,fs)
    if dim == 1
        x[start:stop,:]
    elseif dim ==2
        x[:,start:stop]
    else
        error("Unspported dimension $dim.")
    end
end

function select_bounds(x::MxArray,(start,stop)::Tuple,min,fs,dim)
    start,stop = toindex.((start,stop),min,fs)
    if dim == 1
        mat"x = $x($start:$stop,:);"
    elseif dim == 2
        mat"x = $x(:,$start:$stop);"
    else
        error("Unspported dimension $dim.")
    end

    get_mvariable(:x)
end


function select_bounds(x::AbstractArray,bounds::AbstractArray{<:Tuple},min,fs,dim)
    if dim == 1
        vcat(select_bounds.(Ref(x),bounds,min,fs,dim)...)
    elseif dim == 2
        hcat(select_bounds.(Ref(x),bounds,min,fs,dim)...)
    else
        error("Unspported dimension $dim.")
    end
end

function select_bounds(x::MxArray,bounds::AbstractArray{<:Tuple},min,fs,dim)
    mat"indices = [];"
    for (start,stop) in bounds
        start,stop = toindex.((start,stop),min,fs)
        mat"indices = [indices $start:$stop];"
    end
    if dim == 1
        mat"x = $x(indices,:);"
    elseif dim == 2
        mat"x = $x(:,indices);"
    else
        error("Unspported dimension $dim.")
    end

    get_mvariable(:x)
end
