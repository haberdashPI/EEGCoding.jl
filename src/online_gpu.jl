
function update!(f::Objective{<:CuArray{T}},y,X,λ) where T
    CuArrays.CUBLAS.syrk!('U','T',T(1.0),X,T(λ),f.A) # f.A = λ*f.A + X'X 
    CuArrays.CUBLAS.gemm!('T','N',T(1.0),y,X,T(λ),f.b) # f.b = λ*f.b + y'X
    f
end

function (f::Objective{<:CuArray{T}})(θ,Aθ=BLAS.symm('L','U',f.A,θ)) where T
    # sum(θ'Aθ .- 2.0.*f.b*θ)
    sum(CuArrays.CUBLAS.gemm!('N','N',T(-2.0),f.b,θ,T(1.0),θ'Aθ)) 
end

function ProximalOperators.gradient!(y::CuArray{T},f::Objective{<:CuArray{T}},
    θ::CuArray{T}) where T

    Aθ = CuArrays.CUBLAS.symm!('L','U',T(1.0),f.A,θ,T(0.0),y) 
    f_ = f(θ,Aθ)
    y .= T(2.0).*(Aθ .- f.b')
    f_
end

function code_init(gpu::Val{true},y,X)
    T = Float64
    A = CuArray{T}
    T, A, A(y), A(X)
end


