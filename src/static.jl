export withlags, trf_corr_cv, trf_train, find_envelope

using MetaArrays
using Printf
using DataFrames
using StatsBase
using Statistics
using ShammaModel
using DSP

################################################################################
# testing and training

function trf_train(;prefix,group_suffix="",indices,name="Training",
    progress=Progress(length(indices),1,desc=name),kwds...)

    cachefn(@sprintf("%s_avg%s",prefix,group_suffix),
        trf_train_;prefix=prefix,indices=indices,name=name,progress=progress,
        oncache = () -> ProgressMeter.update!(progress,progress.counter+
            length(indices)),
        kwds...)
end

function trf_train_(;prefix,eeg,lags,indices,stim_fn,name="Training",
        bounds=all_indices,progress=Progress(length(indices),1,desc=name),
        kwds...)

    sum_model = Float64[]

    for i in indices
        stim = stim_fn(i)
        # for now, make signal monaural
        if size(stim,2) > 1
            stim = sum(stim,dims=2)
        end

        model = cachefn(@sprintf("%s_%02d",prefix,i),find_trf,stim,eeg,i,
            -1,lags,"Shrinkage";bounds=bounds[i],kwds...)
        if isempty(sum_model)
            sum_model = model
        else
            sum_model .+= model
        end
        next!(progress)
    end

    sum_model
end

find_envelope(stim,tofs;method=:rms) = find_envelope(stim,tofs,Val(method))

function find_envelope(stim,tofs,::Val{:rms})
    N = round(Int,size(stim,1)/samplerate(stim)*tofs)
    result = zeros(N)
    window_size = 1.5/tofs
    toindex(t) = clamp(round(Int,t*samplerate(stim)),1,size(stim,1))

    for i in 1:N
        t = i/tofs
        from = toindex(t-window_size)
        to = toindex(t+window_size)
        result[i] = mean(x^2 for x in view(stim.data,from:to,:))
    end

    result
end

function find_envelope(stim,tofs,::Val{:audiospect})
    @assert size(stim,2) == 1

    spect_fs = ShammaModel.fixed_fs
    resampled = Filters.resample(vec(stim),spect_fs/samplerate(stim))
    spect = audiospect(SampleBuf(resampled,spect_fs),progressbar=false)
    Filters.resample(vec(sum(spect,dims=2)),tofs/spect_fs)
end
    


find_signals(found_signals,stim,eeg,i;kwds...) = found_signals
function find_signals(::Nothing,stim,eeg,i;bounds=all_indices)
    @assert method == "Shrinkage"
    @assert dir == -1

    response = eegtrial(eeg,i)
    min_len = min(size(stim,1),trunc(Int,size(response,2)));

    stim = select_bounds(stim,bounds,min_len,fs,1)
    response = select_bounds(response,bounds,min_len,fs,2)

    stim,response
end

function find_trf(stim,eeg::MxArray,i,dir,lags,method;found_signals=nothing,
    kwds...)

    @assert method == "Shrinkage"
    @assert dir == -1

    lags = collect(lags)
    mat"$result = FindTRF($stim,$response',-1,[],[],($lags)',$method);"
    result
end

function withlags(x,lags)
    if lags == 0:0
        x
    end

    nl = length(lags)
    n,m = size(x)
    y = similar(x,size(x,1),m*nl)
    z = zero(eltype(y))
    for I in CartesianIndices(x)
        for (l,lag) in enumerate(lags)
            r,c = I[1],I[2]
            r_ = r - lag
            y[r,(l-1)*m+c] = 0 < r_ <= n ? x[r_,c] : z
        end
    end
    y
end

# TODO: we could probably make things even faster if we created the memory XX
# and XY once. 

scale(x) = mapslices(zscore,x,dims=1)
# adds v to the diagonal of matrix (or tensor) x
adddiag!(x,v) = x[CartesianIndex.(axes(x)...)] .+= v
function find_trf(stim,eeg::EEGData,i,dir,lags,method;found_signals=nothing,
    k=0.2,kwds...)

    @assert method == "Shrinkage"
    @assert dir == -1
    stim,response = find_signals(found_signals,stim,eeg,i;kwds...)

    X = withlags(scale(response'),.-reverse(lags))
    Y = scale(stim)

    XX = X'X; XY = Y'X
    λ̄ = tr(XX)/size(X,2)
    XX .*= (1-k); adddiag!(XX,k*λ̄)
    result = XX\XY' # TODO: in Julia 1.2, this can probably be replaced by rdiv!
    reshape(result,size(response,1),length(lags),size(Y,2))
end

function predict_trf(dir,response::MxArray,model,lags,method)
    mat"[~,$result] = FindTRF([],[],-1,$response',$model,($lags)',$method);"
    result
end

function predict_trf(dir,response::Array,model,lags,method)
    @assert method == "Shrinkage"
    @assert dir == -1

    withlags(scale(response'),.-reverse(lags)) * reshape(model,:,size(model,3))
end

function trf_corr_cv(;prefix,indices=indices,group_suffix="",name="Training",
    progress=Progress(length(indices),1,desc=name),kwds...)

    cachefn(@sprintf("%s_corr%s",prefix,group_suffix),
        trf_corr_cv_;prefix=prefix,indices=indices,name=name,progress=progress,
        oncache = () -> update!(progress,progress.counter+length(indices)),
        kwds...)
end

function single(x)
    @assert(length(x) == 1)
    first(x)
end

function trf_corr_cv_(;prefix,eeg,model,lags,indices,stim_fn,
    bounds=all_indices,name="Testing",
    progress=Progress(length(indices),1,desc=name))

    result = zeros(length(indices))

    for (j,i) in enumerate(indices)
        stim = stim_fn(i)
        if size(stim,2) > 1
            stim = sum(stim,dims=2)
        end
        stim,response = find_signals(nothing,stim,eeg,i,
            bounds=bounds[i])

        subj_model_file = joinpath(cache_dir(),@sprintf("%s_%02d",prefix,i))
        # subj_model = load(subj_model_file,"contents")
        subj_model = cachefn(subj_model_file,find_trf,stim,eeg,i,-1,lags,
            "Shrinkage",bounds = bounds[i],
            found_signals = (stim,response))
        n = length(indices)
        r1, r2 = (n-1)/n, 1/n

        pred = predict_trf(-1,response,(r1.*model .- r2.*subj_model),lags,
            "Shrinkage")
        result[j] = single(cor(pred,stim))

        next!(progress)
    end
    result
end