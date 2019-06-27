module EEGCoding
using MATLAB, SampledSignals, Requires

include("parallel.jl")
include("util.jl")
include("eeg.jl")
include("static.jl")
include("online.jl")
@require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
    using .CuArrays
    @info "Loading CuArrays backend for EEGCoding."
    include("online_gpu.jl")
end

end # module
