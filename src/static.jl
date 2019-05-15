export trf_train_speakers

using Printf
using DataFrames
using StatsBase
using Statistics

################################################################################
# testing and training

function trf_train(;prefix,group_suffix="",indices,name="Training",
    progress=Progress(length(indices),1,desc=name),kwds...)

    cachefn(@sprintf("%s_avg%s",prefix,group_suffix),
        trf_train_;prefix=prefix,indices=indices,name=name,progress=progress,
        oncache = () -> update!(progress,progress.counter+length(indices)),
        kwds...)
end

function trf_train_(;prefix,eeg,stim_info,lags,indices,stim_fn,name="Training",
        bounds=all_indices,progress=Progress(length(indices),1,desc=name))
    sum_model = Float64[]

    for i in indices
        stim = stim_fn(i)
        # for now, make signal monaural
        if size(stim,2) > 1
            stim = sum(stim,dims=2)
        end

        model = cachefn(@sprintf("%s_%02d",prefix,i),find_trf,stim,eeg,i,
            -1,lags,"Shrinkage",bounds[i])
        # model = find_trf(stim_envelope,response,-1,lags,"Shrinkage")
        if isempty(sum_model)
            sum_model = model
        else
            sum_model .+= model
        end
        next!(progress)
    end

    sum_model
end

# tried to make things faster by computing envelope in julia (to avoid copying
# the entire wav file): doesn't seem to matter much
function find_envelope(stim,tofs)
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

# function find_envelope(stim,tofs)
#     mat "result = CreateLoudnessFeature($(stim.data),$(samplerate(stim)),$tofs)"
#     get_mvariable(:result)
# end
find_signals(found_signals,stim,eeg,i,bounds=all_indices) = found_signals
function find_signals(::Nothing,stim,eeg,i,bounds=all_indices)
    # envelope and neural response
    fs = samplerate(eeg)
    stim_envelope = find_envelope(stim,fs)

    response = eegtrial(eeg,i)

    min_len = min(size(stim_envelope,1),trunc(Int,size(response,2)));

    stim_envelope = select_bounds(stim_envelope,bounds,min_len,fs,1)
    response = select_bounds(response,bounds,min_len,fs,2)

    stim_envelope,response
end

function find_trf(stim,eeg::MxArray,i,dir,lags,method,bounds=all_indices;
        found_signals=nothing)
    @assert method == "Shrinkage"
    @assert dir == -1
    stim_envelope,response = find_signals(found_signals,stim,eeg,i,bounds)

    lags = collect(lags)
    mat"$result = FindTRF($stim_envelope,$response',-1,[],[],($lags)',$method);"
    result
end

function withlags(x,lags)
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

scale(x) = mapslices(zscore,x,dims=1)
# adds v to the diagonal of matrix (or tensor) x
adddiag!(x,v) = x[CartesianIndex.(axes(x)...)] .+= v
function find_trf(stim,eeg::EEGData,i,dir,lags,method,bounds=all_indices;
        found_signals=nothing,k=0.2)
    @assert method == "Shrinkage"
    @assert dir == -1
    stim_envelope,response = find_signals(found_signals,stim,eeg,i,bounds)

    X = withlags(scale(response'),.-reverse(lags))
    Y = scale(stim_envelope)

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
        trf_corr_cv_,;prefix=prefix,indices=indices,name=name,progress=progress,
        oncache = () -> update!(progress,progress.counter+length(indices)),
        kwds...)
end

function single(x)
    @assert(length(x) == 1)
    first(x)
end

function trf_corr_cv_(;prefix,eeg,stim_info,model,lags,indices,stim_fn,
    bounds=all_indices,name="Testing",
    progress=Progress(length(indices),1,desc=name))

    result = zeros(length(indices))

    for (j,i) in enumerate(indices)
        stim = stim_fn(i)
        if size(stim,2) > 1
            stim = sum(stim,dims=2)
        end
        stim_envelope,response = find_signals(nothing,stim,eeg,i,bounds[i])

        subj_model_file = joinpath(cache_dir,@sprintf("%s_%02d",prefix,i))
        # subj_model = load(subj_model_file,"contents")
        subj_model = cachefn(subj_model_file,find_trf,stim,eeg,i,-1,lags,
            "Shrinkage",bounds[i],found_signals = (stim_envelope,response))
        n = length(indices)
        r1, r2 = (n-1)/n, 1/n

        pred = predict_trf(-1,response,(r1.*model .- r2.*subj_model),lags,
            "Shrinkage")
        result[j] = single(cor(pred,stim_envelope))

        next!(progress)
    end
    result
end

function trf_train_speakers(group_name,files,stim_info;
    skip_bad_trials = false,
    maxlag=0.25,
    train = "" => all_indices,
    test = train)

    train_name, train_fn = train
    test_name, test_fn = test

    df = DataFrame(sid = Int[],condition = String[], speaker = String[],
            corr = Float64[],test_correct = Bool[])

    function setup_indices(events,cond)
        test_bounds = test_fn.(eachrow(events))
        train_bounds = train_fn.(eachrow(events))

        test_indices = findall((events.condition .== cond) .&
            (.!isempty.(test_bounds)) .&
            (.!skip_bad_trials .| .!events.bad_trial))
        train_indices = findall((events.condition .== cond) .&
            (.!isempty.(train_bounds)) .&
            (.!skip_bad_trials .| .!events.bad_trial))

        test_bounds, test_indices, train_bounds, train_indices
    end

    n = 0
    for file in files
        events = events_for_eeg(file,stim_info)[1]
        for cond in unique(events.condition)
            test_bounds, test_indices,
                train_bounds, train_indices = setup_indices(events,cond)
            n += length(train_indices)*3
            n += length(test_indices)*(3+(cond == "male"))
        end
    end
    progress = Progress(n;desc="Analyzing...")

    for file in files
        eeg, stim_events, sid = load_subject(joinpath(data_dir,file),stim_info)
        lags = 0:round(Int,maxlag*samplerate(eeg))
        sid_str = @sprintf("%03d",sid)

        target_len = convert(Float64,stim_info["target_len"])

        for cond in unique(stim_events.condition)
            test_bounds, test_indices,
             train_bounds, train_indices = setup_indices(stim_events,cond)

            for (speaker_index,speaker) in enumerate(["male", "fem1", "fem2"])

                prefix = join([train_name,"trf",cond,speaker,sid_str],"_")
                model = trf_train(
                    prefix = prefix,
                    eeg = eeg,
                    stim_info = stim_info,lags=lags,
                    indices = train_indices,
                    group_suffix = "_"*group_name,
                    bounds = train_bounds,
                    progress = progress,
                    stim_fn = i -> load_sentence(stim_events,stim_info,i,
                        speaker_index)
                )

                prefix = join([test_name,"trf",cond,speaker,sid_str],"_")
                C = trf_corr_cv(
                    prefix=prefix,
                    eeg=eeg,
                    stim_info=stim_info,
                    model=model,
                    lags=lags,
                    indices = test_indices,
                    group_suffix = "_"*group_name,
                    bounds = test_bounds,
                    progress = progress,
                    stim_fn = i -> load_sentence(stim_events,stim_info,i,
                        speaker_index)
                )
                rows = DataFrame(
                    sid = sid,
                    condition = cond,
                    speaker=speaker,
                    corr = C,
                    test_correct = stim_events.correct[test_indices]
                )
                df = vcat(df,rows)

                if speaker == "male"
                    prefix = join([test_name,"trf",cond,"male_other",sid_str],"_")
                    C = trf_corr_cv(
                        prefix=prefix,
                        eeg=eeg,
                        stim_info=stim_info,
                        model=model,
                        lags=lags,
                        indices = test_indices,
                        group_suffix = "_"*group_name,
                        bounds = test_bounds,
                        progress = progress,
                        stim_fn = i -> load_other_sentence(stim_events,stim_info,i,1)
                    )
                    rows = DataFrame(
                        sid = sid,
                        condition = cond,
                        speaker="male_other",
                        corr = C,
                        test_correct = stim_events.correct[test_indices]
                    )
                    df = vcat(df,rows)

                end
            end
        end

    end

    df
end
