struct SpectralParameters{T}
    # plan::Base.DFT.FFTW.cFFTWPlan{Complex{T},-1,true,1}
    # iplan::Base.DFT.ScaledPlan{Complex{T},Base.DFT.FFTW.cFFTWPlan{Complex{T},1,true,1},T}
    plan
    iplan
    A::T
    B::Vector{T}
    force::T
    vv::Vector{Complex{T}}
    T0::T
end

"""
    conv(p,ip,u,v)

Convolution of two vectors. Uses a planned FFT algorithm.

This function directly borrows from the conv in DSP.jl:
https://github.com/JuliaDSP/DSP.jl/blob/master/src/dspbase.jl
"""
function DSP.conv(p,ip,u,v)
    T = eltype(u)
    nu = length(u)
    nv = length(v)
    n = nu+nv-1
    np2 = n>1024 ? nextprod([2,3,5],n) : nextpow(2,n)
    upad = [u;zeros(T,np2-nu)]
    vpad = [v;zeros(T,np2-nv)]

    y = ip*((p*upad).*(p*vpad))
    return y[1:n]
end

"""
    conv_mat(u)
Calculate a matrix `A` such that `A*v = conv(u,v)`.
"""
function conv_mat(u)
    nn = length(u)
    nd2 = round(Int,nn/2)
    inds = (-nd2:nd2-1).+nn.+1

    δ = zeros(eltype(u),nn)
    δ[1] = one(eltype(u))
    v1 = conv(δ,u)[inds][1:end]

    δ *= zero(eltype(u))
    δ[end] = one(eltype(u))
    v2 = conv(δ,u)[inds][end:-1:1]

    v1[1] = v2[1]

    ToeplitzMatrices.Toeplitz(v1,v2)
end

"""
    conv_mat(p,ip,u)
Same as `conv_mat(u)` except it reqires a forward plan `p` and an inverse plan `ip`.
Both plans need to be appropriately sized so that `p*conv(u,u)` will work.
"""
function conv_mat(p,ip,u)
    nn = length(u)
    nd2 = round(Int,nn/2)
    inds = (-nd2:nd2-1).+nn.+1

    δ = zeros(eltype(u),nn)
    δ[1] = one(eltype(u))
    v1 = conv(p,ip,δ,u)[inds][1:end]

    δ *= zero(eltype(u))
    δ[end] = one(eltype(u))
    v2 = conv(p,ip,δ,u)[inds][end:-1:1]

    v1[1] = v2[1]

    ToeplitzMatrices.Toeplitz(v1,v2)
end

function spectral_rhs!(::Type{Val{:jac}},jac,ϕ,params,t)
    nn = length(params.vv)
    p = view(ϕ,1:nn)
    tt = view(ϕ,nn+1:2nn)

    density_pjac!(view(jac,1:nn,1:nn),p,tt,params)
    density_ttjac!(view(jac,1:nn,nn+1:2nn),p,tt,params)
    temperature_pjac!(view(jac,nn+1:2nn,1:nn),tt,p,params)
    temperature_ttjac!(view(jac,nn+1:2nn,nn+1:2nn),tt,p,params)
    jac
end

function density!(dp,p,tt,params)
    @unpack plan,iplan,A,B,force,vv,T0 = params
    nn = length(axes(p)[1])
    nd2 = round(Int,nn/2)
    kk = -nd2:nd2-1
    dp[:] .= (-im*2π*force*kk.*p
              .-4π^2*( conv(plan,iplan,p.*kk.^2,tt)[kk.+nn.+1]
                      .+conv(plan,iplan,p.*kk,tt.*kk)[kk.+nn.+1])
              .-4π^2*( conv(plan,iplan,p,vv.*kk.^2)[kk.+nn.+1]
                      .+conv(plan,iplan,p.*kk,vv.*kk)[kk.+nn.+1]))
end

function density_pjac!(pjac,p,tt,params)
    @unpack plan,iplan,A,B,force,vv,T0 = params
    nn = length(axes(p)[1])
    nd2 = round(Int,nn/2)
    kk = (-nd2:nd2-1)*one(eltype(p)) |> collect

    K = Diagonal(kk)
    pjac[:,:] .= -im*2π*force*K.-4π^2*(  conv_mat(plan,iplan,tt)*K.^2 .+conv_mat(plan,iplan,tt.*kk)*K
                                       .+conv_mat(plan,iplan,vv.*kk.^2).+conv_mat(plan,iplan,vv.*kk)*K)
end

function density_ttjac!(ttjac,p,tt,params)
    @unpack plan,iplan,A,B,force,vv,T0 = params
    nn = length(axes(p)[1])
    nd2 = round(Int,nn/2)
    kk = (-nd2:nd2-1)*one(eltype(p)) |> collect

    K = Diagonal(kk)
    ttjac[:,:] .= -4π^2*(conv_mat(plan,iplan,p.*kk.^2).+conv_mat(plan,iplan,p.*kk)*K)
end

function temperature!(dtt,tt,p,params::SpectralParameters{T}) where T <: Number
    @unpack plan,iplan,A,B,force,vv,T0 = params
    nn = length(axes(p)[1])
    nd2 = round(Int,nn/2)
    kk = -nd2:nd2-1
    inds = (-nd2:nd2-1).+nn.+1
    delta_k_0 = zeros(nn)
    delta_k_0[nd2+1] = T0
    dtt[:] .= ( A*force^2*p-2π*im*A*force*conv(plan,iplan,kk.*p,tt)[kk.+nn.+1]
               -4π*im*A*force*conv(plan,iplan,p,kk.*vv)[kk.+nn.+1]
               -4π^2*A*conv(plan,iplan,tt,conv(plan,iplan,kk.*p,kk.*vv)[inds])[inds]
               -4π^2*A*conv(plan,iplan,p,conv(plan,iplan,vv.*kk,vv.*kk)[inds])[inds]
               -4π^2*B[1]*tt.*(kk.^2)
               +B[2]*(delta_k_0-tt))
end

function temperature_pjac!(pjac,tt,p,params)
    @unpack plan,iplan,A,B,force,vv,T0 = params
    nn = length(axes(p)[1])
    nd2 = round(Int,nn/2)
    kk = (-nd2:nd2-1)*one(eltype(p)) |> collect
    inds = (-nd2:nd2-1).+nn.+1

    K = Diagonal(kk)
    pjac[:,:] .= (A*force^2*Diagonal(ones(nn)).-2π*im*A*force*conv_mat(plan,iplan,tt)*K
                  .-4π*im*A*force*conv_mat(plan,iplan,vv.*kk)
                  .-4π^2*A*(  (conv_mat(plan,iplan,conv(plan,iplan,tt,kk.*vv)[inds]))*K
                            .+conv_mat(plan,iplan,conv(plan,iplan,vv.*kk,vv.*kk)[inds]))
    )
end

function temperature_ttjac!(ttjac,tt,p,params)
    @unpack plan,iplan,A,B,force,vv,T0 = params
    nn = length(axes(p)[1])
    nd2 = round(Int,nn/2)
    kk = (-nd2:nd2-1)*one(eltype(p)) |> collect
    inds = (-nd2:nd2-1).+nn.+1

    K = Diagonal(kk)
    ttjac[:,:] .= (-2π*im*A*force*conv_mat(kk.*p)
                   .-4π^2*A*conv_mat(plan,iplan,conv(plan,iplan,kk.*p,kk.*vv)[inds])
                   .-4π^2*B[1]*K.^2
                   .-B[2]*Diagonal(ones(nn)))
end

function spectral_rhs!(dϕ,ϕ,params,t)
    @unpack plan,iplan,A,B,force,vv,T0 = params
    nn = round(Int,length(ϕ)/2)
    dϕ[1:nn] .= density!(dϕ[1:nn],ϕ[1:nn],ϕ[nn+1:2nn],params)
    dϕ[nn+1:2nn] .= temperature!(dϕ[nn+1:2nn],ϕ[nn+1:2nn],ϕ[1:nn],params)
    dϕ
end

function real_current(ϕ,params)
    nn = round(Int,length(ϕ)/2)
    nd2 = round(Int,nn/2)
    kk = -nd2:nd2-1
    p = ϕ[1:nn]
    tt = ϕ[nn+1:2nn]

    ∇p = -im*2π*kk.*p
    ∇tt = -im*2π*kk.*tt
    ∇vv = -im*2π*kk.*params.vv

    P = fft(fftshift(p))
    T = fft(fftshift(tt))
    ∇P = fft(fftshift(∇p))
    ∇T = fft(fftshift(∇tt))
    ∇V = fft(fftshift(∇vv))

    J = -(P.*(∇V.+params.force).+T.*∇P)
end
