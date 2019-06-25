module EEGCoding
using MATLAB, SampledSignals

include("parallel.jl")
include("util.jl")
include("eeg.jl")
include("static.jl")
include("online.jl")

end # module
