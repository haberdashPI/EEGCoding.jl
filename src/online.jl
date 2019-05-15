using ProximalOperators
using ProximalAlgorithms
using Unitful
using SampledSignals
using LinearAlgebra
using ProgressMeter

# TODO: preprcoessing, add lag and add intercept

################################################################################
"""
    code(y,X,[state];params...)

Solve single step of online encoding or decoding problem. For decoding, y =
speech stimulus, and X = eeg data. For encoding, y = eeg data, X = speech
stimulus. You should pass a single window of data to the function, with rows
representing time slices, and columns representing channels (for eeg data) or
features (for stimulus data).

The coding coefficients are computed according to the following optimization
problem

```math
\\underset{\\theta}{\\mathrm{arg\\ min}} \\quad \\sum_i
    \\lambda^{k-i} \\left\\lVert y_i - X_i\\theta \\right\\rVert^2 +
    \\gamma\\left\\lVert\\theta\\right\\rVert
```

In the above equation, y is the output, X the input, and θ the parameters to
be solved for.

Returns an `Objective` object (see below).
"""
function code
end # see full defintion below

# defines the objective of optimization
struct Objective <: ProximalOperators.Quadratic
    A::Symmetric{Float64,Matrix{Float64}}
    b::Matrix{Float64}
    θ::Matrix{Float64}
    function Objective(y::Union{AbstractVector,AbstractMatrix},
        X::AbstractMatrix)

        new(
            Symmetric(zeros(size(X,2),size(X,2))),
            zeros(size(y,2),size(X,2)),
            zeros(size(X,2),size(y,2))
        )
    end
end
ProximalOperators.fun_dom(f::Objective) = "AbstractArray{Real}"
ProximalOperators.fun_expr(f::Objective) = "x ↦ x'Ax - 2bx"
ProximalOperators.fun_params(f::Objective) = "" # parameters will be too large...

function update!(f::Objective,y,X,λ)
    BLAS.syrk!('U','T',1.0,X,λ,f.A.data) # f.A = λ*f.A + X'X 
    BLAS.gemm!('T','N',1.0,y,X,λ,f.b) # f.b = λ*f.b + y'X
    f
end

function (f::Objective)(θ)
    y = θ'f.A*θ
    BLAS.gemm!('N','N',-2.0,f.b,θ,1.0,y) # y .-= 2.0.*f.b*θ
    sum(y)
end

function ProximalOperators.gradient!(y::AbstractArray,f::Objective,x::AbstractArray)
    y .= 2.0.*f.A*x .- 2.0.*f.b'
    f(x)
end

function code(y,X,state=nothing;λ=(1-1/30),γ=1e-3,kwds...)
    state = isnothing(state) ? Objective(y,X) : state
    params = merge((maxit=1000, tol=1e-3, fast=true, verbose=0),kwds)

    update!(state,y,X,λ)
    _, result = ProximalAlgorithms.FBS(state.θ,fs=state,fq=NormL1(γ);params...)
    state.θ .= result

    state
end

################################################################################
"""
    marker(eeg,targets...;params)

Computes an attentional marker for each specified target, using the L1 norm
of the online decoding coefficients.
"""
function marker
end

asframes(x,signal) = asseconds(x)*samplerate(signal)
asseconds(x) = x
asseconds(x::Quantity) = ustrip(uconvert(s,x))

function marker(eeg,targets...;
    # marker parameters
    window=250ms,
    lag=400ms,
    estimation_length=5s,
    min_norm=1e-4,
    code_params...)

    # TODO: this thing about the lag doesn't actually make much sense ... I
    # think that's for compensating for something I'm ultimately not doing
    # for now I'll leave it to compare results to the matlab code

    nt = length(targets)
    nlag = floor(Int,asframes(lag,eeg))+1
    ntime = size(eeg,1)-nlag+1
    window_len = floor(Int,asframes(window,eeg))
    nwindows = div(ntime,window_len)
    λ = 1 - window_len/(asframes(estimation_length,eeg))

    results = map(_ -> Array{Float64}(undef,nwindows),targets)

    atwindow(x;length,index) = x[(1:length) .+ (index-1)*length,:]

    states = fill(nothing,nt)
    @showprogress for w in 1:nwindows
        # TODO: decoding might work better if we allow for an intercept
        # at each step
        windows = atwindow.((eeg,targets...),length=window_len,index=w)
        eegw = windows[1]
        states = map(windows[2:end],states,1:nt) do targetw,state,ti
            state = code(targetw,eegw,state;λ=λ,code_params...)
            results[ti][w] = max(norm(state.θ,1), min_norm)

            state
        end
    end

    results
end

################################################################################
"""
    attention(x,y)

Given two attention markers, x and y, use a batch, probabilistic state space
model to compute a smoothed attentional state for x.
"""
function attention(x1,x2;
    outer_EM_iter = 20,
    inner_EM_iter = 20,
    newton_iter = 10,

    # inverse-gamma prior
    μ_p = 0.2,
    σ_p = 5,
    a₀ = 2+μ_p^2/σ_p,
    b₀ = μ_p*(a₀-1),

    # prior of attended (i = 1) and unattended (i = 2) sources
    μ_d₀ = [0.88897,0.16195],
    ρ_d₀ = [16.3344,8.2734],
    α₀ = [5.7111e+02,1.7725e+03],
    β₀ = [34.96324,2.1424e+02],
    μ₀ = [0.88897,0.1619])

    n = size(x1,1)

    # Kalman filtering and smoothing variables
    z_t = zeros(n+1,1); z_t₁ = zeros(n+1,1); z_T = zeros(n+1,1);
    σ_t = zeros(n+1,1); σ_t₁ = zeros(n+1,1); σ_T = zeros(n+1,1);
    S = zeros(n,1);

    # batch state-space model outputs for the two attentional markers
    z = zeros(n,1);
    η = 0.3ones(n,1);

    ρ_d = ρ_d₀
    μ_d = μ_d₀

    P(x,i) = (1/x)*sqrt(ρ_d[i])*exp(-0.5ρ_d[i]*(log(x)-μ_d[i])^2)
    function outerE(x1,x2,z)
        p = 1/(1+exp(-z))
        p*P(x1,1)*P(x2,1) /
            (p*P(x1,1)*P(x2,2) + (1-p)*(P(x1,2)*P(x2,2)))
    end
    function outerM(E,x1,x2,i)
        μ = ( sum(E.*log.(x1)+(1-E).*log.(x2)) + n*μ₀[i] )/2n
        ρ = (2n*α₀[i])/(
            sum( E.*((log.(x1).-μ_d[i]).^2) .+
                 (1.0-E).*((log.(x2)-μ_d[i]).^2) ) +
            n*(2*β₀[i]+(μ_d[i]-μ₀[i]).^2)
        )
        μ,ρ
    end

    ########################################
    # outer EM
    for i in 1:outer_EM_iter
        # calculating epsilon_k's in the current iteration (E-Step)
        E = outerE.(x1,x2,z)

        # prior update
        μ_d[1], ρ_d[1] = outerM(E,x1,x2,1)
        μ_d[2], ρ_d[2] = outerM(E,x1,x2,2)

        ########################################
        # inner EM for updating z's and η's (M-Step)
        for j in 1:inner_EM_iter
            ##############################
            # inner E

            # filtering
            for t in 2:n+1
                z_t₁[t] = z_t[t-1];
                σ_t₁[t] = σ_t[t-1] + η[t-1];

                # Newton's algorithm
                for m=1:newton_iter
                    a = z_t[t] - z_t₁[t] - σ_t₁[t]*(E[t-1] -
                        exp(z_t[t])/(1+exp(z_t[t])))
                    b = 1 + σ_t₁[t]*exp(z_t[t])/((1+exp(z_t[t]))^2)
                    z_t[t] -= a / b;
                end
                σ_t[t] = 1 / (1/σ_t₁[t] +
                    exp(z_t[t])/((1+exp(z_t[t]))^2));
            end

            # smoothing
            z_T[end] = z_t[n+1];
            σ_T[n+1] = σ_T[n+1];

            for t = reverse(1:n)
                S[t] = σ_t[t]*1/σ_t₁[t+1];
                z_T[t] = z_t[t] + S[t]*(z_T[t+1] - z_t₁[t+1])
                σ_T[t] = σ_t[t] + S[t]^2*(σ_T[t+1] -
                    σ_t₁[t+1]);
            end

            z_t[1] = z_T[1]
            σ_t[1] = σ_T[1]

            ####################
            # inner M (update z and η)
            η = ((z_T[2:end]-z_T[1:end-1]).^2 + σ_T[2:end] +
                 σ_T[1:end-1]-2*σ_T[2:end].*S + 2*b₀) / (1+2*(a₀+1));
        end

        # update the z's
        z = z_T[2:end];
    end

end
