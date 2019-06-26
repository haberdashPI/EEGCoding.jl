using Distributed, ProgressMeter, Match
export parallel_progress, progress_update!

abstract type ProgressMessage end
isdone(x::ProgressMessage) = false
struct ProgressAmmendTotal <: ProgressMessage
    x::Int
end
struct ProgressIncrement <: ProgressMessage
    x::Int
end
struct ProgressDone <: ProgressMessage end
isdone(x::ProgressDone) = true

const ProgressChannel = RemoteChannel{Channel{ProgressMessage}}

progress_update!(prog::ProgressChannel,n=1) = put!(prog,ProgressIncrement(n))
function progress_ammend!(prog::ProgressChannel,n)
    put!(prog,ProgressAmmendTotal(n))
    # yield()
end

parallel_progress(fn,n::Number) = parallel_progress(fn,Progress(n))
function parallel_progress(fn,ch::ProgressChannel) 
    # @info "already created progress channel"
    fn(ch)
end
parallel_progress(fn,prog::Bool) = 
    prog ? parallel_progress(fn,Progress(0)) : fn(prog)
parallel_progress(fn,n::Number,prog::Bool) =
    prog ? parallel_progress(fn,Progress(n)) : fn(prog)
function parallel_progress(fn,n::Number,ch::Union{Progress,ProgressChannel})
    progress_ammend!(ch,n)
    fn(ch)
end

function parallel_progress(fn,progress::Progress)
    # @info "setting up progress channel"
    # fn(progress)
    channel = RemoteChannel(()->Channel{ProgressMessage}(2^8),1)
    result = nothing
    @sync begin
        @async begin
            update = take!(channel)
            while true
                @match update begin
                    ProgressDone() => break
                    ProgressIncrement(n) => progress_update!(progress,n)
                    ProgressAmmendTotal(n) => progress_ammend!(progress,n)
                end
                update = take!(channel)
            end
        end

        @async begin
            result = fn(channel)
            put!(channel,ProgressDone())
        end
    end
    result
end
