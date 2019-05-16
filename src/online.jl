using ProximalOperators
using ProximalAlgorithms
using Unitful
using SampledSignals
using LinearAlgebra
using ProgressMeter
using Distributions

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
function attention(m1,m2;
    outer_EM_iter = 20,
    inner_EM_iter = 20,
    newton_iter = 10,
    ci = 0.9, # for #90 confidence intervals

    mean_p = 0.2,
    var_p = 5,
    a₀ = 2+mean_p^2/var_p,
    b₀ = mean_p*(a₀-1),

    # tuned attended and unattended prior distributions based on a similar trial
    α₀ = [5.7111e+02,1.7725e+03],
    β₀ = [34.96324,2.1424e+02],
    μ₀ = [0.88897,0.1619],

    # initialized attended and unattended Log-Normal distribution parameters
    # based on a similar trial
    ρ_d₀ = [16.3344,8.2734],
    μ_d₀ = [0.88897,0.16195])

    n = min(size(m1,1),size(m2,1))

    # Kalman filtering and smoothing variables
    z_k = zeros(n+1)
    z_k₁ = zeros(n+1)
    σ_k = zeros(n+1)
    σ_k₁ = zeros(n+1)

    z_K = zeros(n+1)
    σ_K = zeros(n+1)

    S = zeros(n)

    ρ_d = ρ_d₀
    μ_d = μ_d₀

    # final results
    z = zeros(n)
    η = fill(0.3,n)

    P(m,i) = (1/m)*sqrt(ρ_d[i])*exp(-0.5*ρ_d[i]*(log(m)-μ_d[i])^2)
    function outerE(m1,m2,z)
        p = 1/(1+exp(-z))
        P1_a = P(m1,1); P1_u = P(m1,2) 
        P2_u = P(m2,2); P2_a = P(m2,1) 
        (p*P1_a*P2_u)/(p*P1_a*P2_u+(1-p)*P1_u*P2_a)
    end

    function outerM(E,ma,mb,i)
        μ = ( sum(E.*log.(ma).+(1.0.-E).*log.(mb)) + n*μ₀[i] )/2n
        s = sum( E.*((log.(ma).-μ_d[i]).^2) .+ 
                 (1.0.-E).*((log.(mb).-μ_d[i]).^2) )
        ρ = (2n*α₀[i])/( s + n*(2*β₀[i]+(μ_d[i]-μ₀[i]).^2) )

        μ,ρ
    end

    # outer EM
    for i in 1:outer_EM_iter
        E = outerE.(m1,m2,z)
        μ_d[1], ρ_d[1] = outerM(E,m1,m2,1)
        μ_d[2], ρ_d[2] = outerM(E,m2,m1,2)

        # inner EM for updating z's and eta's (M-Step)
                
        for j in 1:inner_EM_iter

            # Filtering
            for k in 2:n+1
                z_k₁[k] = z_k[k-1]
                σ_k₁[k] = σ_k[k-1] + η[k-1]

                # Newton's Algorithm
                for m in 1:newton_iter
                    a = z_k[k] - z_k₁[k] - σ_k₁[k]*(E[k-1] - 
                        exp(z_k[k])/(1+exp(z_k[k])))
                    b = 1 + σ_k₁[k]*exp(z_k[k])/((1+exp(z_k[k]))^2)
                    z_k[k] = z_k[k] -  a/b 
                end

                σ_k[k] = 1 / (1/σ_k₁[k] + exp(z_k[k])/((1+exp(z_k[k]))^2))
            end

            # Smoothing
            z_K[n+1] = z_k[n+1]
            σ_K[n+1] = σ_K[n+1]

            for k = reverse(1:n)
                S[k] = σ_k[k]/σ_k₁[k+1]
                z_K[k] = z_k[k] + S[k]*(z_K[k+1] - z_k₁[k+1])
                σ_K[k] = σ_k[k] + S[k]^2*(σ_K[k+1] - σ_k₁[k+1])

            end
            z_k[1] = z_K[1]
            σ_k[1] = σ_K[1]

            # update the eta's
            η = ((z_K[2:end].-z_K[1:end-1]).^2 + 
                  σ_K[2:end] .+ σ_K[1:end-1] .- 2.0.*σ_K[2:end].*S .+ 2.0.*b₀) / 
                (1.0.+2.0.*(a₀.+1.0))
        end
        
        # update the z's
        z = z_K[2:end]
    end

    scale = cquantile(Normal(),0.5(1-ci))
    z, z .+ scale.*sqrt.(η), z .- scale.*sqrt.(η)
end