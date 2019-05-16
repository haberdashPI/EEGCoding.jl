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
function attention(m1,m2;
    outer_EM_batch = 20,
    inner_EM_batch = 20,
    Newton_iter = 10,
    c0 = 1.65, # for #90 confidence intervals

    mean_p = 0.2,
    var_p = 5,
    a_0 = 2+mean_p^2/var_p,
    b_0 = mean_p*(a_0-1))

    n = min(size(m1,1),size(m2,1))

    # Kalman filtering and smoothing variables
    z_k = zeros(n+1,1)
    z_k_1 = zeros(n+1,1)
    sig_k = zeros(n+1,1)
    sig_k_1 = zeros(n+1,1)

    z_K = zeros(n+1,1)
    sig_K = zeros(n+1,1)

    S = zeros(n,1)

    # tuned attended and unattended prior distributions based on a similar trial
    alpha_0 = [5.7111e+02,1.7725e+03]
    beta_0 = [34.96324,2.1424e+02]
    mu_0 = [0.88897,0.1619]

    # initialized attended and unattended Log-Normal distribution parameters
    # based on a similar trial
    rho_d_0 = [16.3344,8.2734]
    mu_d_0 = [0.88897,0.16195]

    rho_d = rho_d_0
    mu_d = mu_d_0

    # batch state-space model outputs for the two attentional markers
    z_batch = zeros(n,1)
    eta_batch = 0.3*ones(n,1)

    # outer EM
    for i in 1:outer_EM_batch
        # calculating epsilon_k's in the current iteration (E-Step)
                

        P_11 = @. (1/m1)*sqrt(rho_d[1])*exp(-0.5*rho_d[1]*(log(m1)-mu_d[1])^2)
        P_12 = @. (1/m1)*sqrt(rho_d[2]).*exp(-0.5*rho_d[2]*(log(m1)-mu_d[2])^2)
        P_21 = @. (1/m2)*sqrt(rho_d[2]).*exp(-0.5*rho_d[2]*(log(m2)-mu_d[2])^2)
        P_22 = @. (1/m2)*sqrt(rho_d[1]).*exp(-0.5*rho_d[1]*(log(m2)-mu_d[1])^2)

        p = @. (1/(1+exp(-z_batch)))
        
        E = @. (p*P_11*P_21)/(p*P_11*P_21+(1-p)*P_12*P_22)
                
        # prior update
        mu_d[1] = ( sum(E.*log.(m1).+(1.0.-E).*log.(m2)) + n*mu_0[1] )/(2*n)
        mu_d[2] = ( sum(E.*log.(m2).+(1.0.-E).*log.(m1)) + n*mu_0[2] )/(2*n)
        rho_d[1] = (2*n*alpha_0[1])/( sum( E.*((log.(m1).-mu_d[1]).^2) .+ (1.0.-E).*((log.(m2).-mu_d[1]).^2) ) + n*(2*beta_0[1]+(mu_d[1]-mu_0[1]).^2) )
        rho_d[2] = (2*n*alpha_0[2])/( sum( E.*((log.(m2).-mu_d[2]).^2) .+ (1.0.-E).*((log.(m1).-mu_d[2]).^2) ) + n*(2*beta_0[2]+(mu_d[2]-mu_0[2]).^2) )

        # inner EM for updating z's and eta's (M-Step)
                
        for j in 1:inner_EM_batch

            # Filtering
            for kk in 2:n+1
                z_k_1[kk] = z_k[kk-1]
                sig_k_1[kk] = sig_k[kk-1] + eta_batch[kk-1]

                # Newton's Algorithm
                for m in 1:Newton_iter
                    z_k[kk] = z_k[kk] - (z_k[kk] - z_k_1[kk] - sig_k_1[kk]*(E[kk-1] - exp(z_k[kk])/(1+exp(z_k[kk])))) / (1 + sig_k_1[kk]*exp(z_k[kk])/((1+exp(z_k[kk]))^2))
                end

                sig_k[kk] = 1 / (1/sig_k_1[kk] + exp(z_k[kk])/((1+exp(z_k[kk]))^2))
            end

            # Smoothing
            z_K[n+1] = z_k[n+1]
            sig_K[n+1] = sig_K[n+1]

            for kk = n:-1:1
                S[kk] = sig_k[kk]/sig_k_1[kk+1]
                z_K[kk] = z_k[kk] + S[kk]*(z_K[kk+1] - z_k_1[kk+1])
                sig_K[kk] = sig_k[kk] + S[kk]^2*(sig_K[kk+1] - sig_k_1[kk+1])

            end


            z_k[1] = z_K[1]
            sig_k[1] = sig_K[1]

            # update the eta's
            eta_batch = @.( (z_K[2:end]-z_K[1:end-1])^2+sig_K[2:end]+sig_K[1:end-1]-2*sig_K[2:end]*S+2b_0 )/(1+2*(a_0+1))

        end
        
        # update the z's
        z_batch = z_K[2:end]
    end

    z_batch, eta_batch
end
