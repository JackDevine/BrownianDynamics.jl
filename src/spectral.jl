function conv2!(p,ip,At::StridedMatrix{T},Bt::StridedMatrix{T},C::StridedMatrix{T},
                A::StridedMatrix{T},B::StridedMatrix{T}) where T <: Complex
    sa,sb = size(A),size(B)

    At[:,:] = zero(T)
    Bt[:,:] = zero(T)
    At[1:sa[1],1:sa[2]] .= A
    Bt[1:sb[1],1:sb[2]] .= B

    A_mul_B!(C,ip,(p*At).*(p*Bt))
    return C
end
conv2!(A::StridedMatrix{T},B::StridedMatrix{T}) where {T<:Integer} =
       round.(Int,conv2(float(A), float(B)))
conv2!(u::StridedVector{T},v::StridedVector{T},A::StridedMatrix{T}) where {T<:Integer} =
       round.(Int, conv2(float(u),float(v),float(A)))


function density!(dp,p,tt,params)
    plan,iplan,At,Bt,C,vv,f = getindex.(params,[:plan,:iplan,:At,:Bt,:C,:vv,:f])
    nn = size(p)[1]
    kk = repmat(-round(Int, nn/2):round(Int, nn/2-1),1,nn)
    ll = copy(kk)'
    inds = (-round(Int,nn/2):round(Int,nn/2-1))+nn+1
    dp[:,:] .= (-im*2π*(f[1]*kk.+f[2]*ll).*p
                 .- 4π^2*(  conv2!(plan,iplan,At,Bt,C,p.*(kk.^2.+ll.^2),tt)[inds,inds]
                          .+conv2!(plan,iplan,At,Bt,C,p.*kk,tt.*kk)[inds,inds]
                          .+conv2!(plan,iplan,At,Bt,C,p.*ll,tt.*ll)[inds,inds])
                 .-4π^2*(  conv2!(plan,iplan,At,Bt,C,p,vv.*(kk.^2.+ll.^2))[inds,inds]
                         .+conv2!(plan,iplan,At,Bt,C,p.*kk,vv.*kk)[inds,inds]
                         .+conv2!(plan,iplan,At,Bt,C,p.*ll,vv.*ll)[inds,inds]))
    dp
end


function matching_indices(current_indices,center_indices)
    current_indices[[in(index, center_indices) for index in current_indices]]
end

function density_pjac!(pjac,p,tt,params)
    plan,iplan,At,Bt,C,vv,f = getindex.(params,[:plan,:iplan,:At,:Bt,:C,:vv,:f])
    nn = size(p)[1]
    nd2 = round(Int,nn/2)
    nd4 = round(Int,nn/4)
    kk = repmat((-nd2:nd2-1)*(1+0im),1,nn)
    ll = copy(kk)'
    inds = (-nd2:(nd2-1))+nn+1

    pjac[:,:] .= -im*2π*Diagonal((f[1]*kk.+f[2]*ll)[:])
    δ_i_j = zeros(eltype(p),nn,nn)
    for j in 1:nn^2
        δ_i_j *= zero(eltype(p))
        δ_i_j[j] = one(eltype(p))

        pjac[:,j] .+= (.-4π^2*(  conv2!(plan,iplan,At,Bt,C,δ_i_j.*(kk.^2 .+ll.^2),tt)[inds,inds]
                                .+conv2!(plan,iplan,At,Bt,C,δ_i_j.*kk,tt.*kk)[inds,inds]
                                .+conv2!(plan,iplan,At,Bt,C,δ_i_j.*ll,tt.*ll)[inds,inds])
                       .-4π^2*(  conv2!(plan,iplan,At,Bt,C,δ_i_j,vv.*(kk.^2 .+ll.^2))[inds,inds]
                               .+conv2!(plan,iplan,At,Bt,C,δ_i_j.*kk,vv.*kk)[inds,inds]
                               .+conv2!(plan,iplan,At,Bt,C,δ_i_j.*ll,vv.*ll)[inds,inds]))[:]
    end
    pjac
end

function density_pjac_approx!(pjac,p,tt,params,filt_matrix)
    plan,iplan,At,Bt,C,vv,f = getindex.(params,[:plan,:iplan,:At,:Bt,:C,:vv,:f])
    nn = size(p)[1]
    nd2 = round(Int,nn/2)
    nd4 = round(Int,nn/4)
    kk = repmat((-nd2:nd2-1)*(1+0im),1,nn)
    ll = copy(kk)'
    inds = (-nd2:(nd2-1))+nn+1

    pjac[:,:] .= -im*2π*Diagonal((f[1]*kk.+f[2]*ll)[:])

    δ = zeros(eltype(p),nn,nn)
    δ[1] = 1
    v1 = (.-4π^2*(  conv2!(plan,iplan,At,Bt,C,δ.*(kk.^2 .+ll.^2),tt)[inds,inds]
                                .+conv2!(plan,iplan,At,Bt,C,δ.*kk,tt.*kk)[inds,inds]
                                .+conv2!(plan,iplan,At,Bt,C,δ.*ll,tt.*ll)[inds,inds])
                       .-4π^2*(  conv2!(plan,iplan,At,Bt,C,δ,vv.*(kk.^2 .+ll.^2))[inds,inds]
                               .+conv2!(plan,iplan,At,Bt,C,δ.*kk,vv.*kk)[inds,inds]
                               .+conv2!(plan,iplan,At,Bt,C,δ.*ll,vv.*ll)[inds,inds]))[:]

    δ[1] = zero(eltype(p))
    δ[end] = 1
    v2 = (.-4π^2*(  conv2!(plan,iplan,At,Bt,C,δ.*(kk.^2 .+ll.^2),tt)[inds,inds]
                                .+conv2!(plan,iplan,At,Bt,C,δ.*kk,tt.*kk)[inds,inds]
                                .+conv2!(plan,iplan,At,Bt,C,δ.*ll,tt.*ll)[inds,inds])
                       .-4π^2*(  conv2!(plan,iplan,At,Bt,C,δ,vv.*(kk.^2 .+ll.^2))[inds,inds]
                               .+conv2!(plan,iplan,At,Bt,C,δ.*kk,vv.*kk)[inds,inds]
                               .+conv2!(plan,iplan,At,Bt,C,δ.*ll,vv.*ll)[inds,inds]))[:]
    v2[1] = v1[1]

    pjac[:,:] .+= (filt_matrix.*ToeplitzMatrices.Toeplitz(v1,v2))[:,:]
 end

function density_ttjac!(ttjac,p,tt,params)
    plan,iplan,At,Bt,C,vv,f = getindex.(params,[:plan,:iplan,:At,:Bt,:C,:vv,:f])
    nn = size(p)[1]
    kk = repmat(-round(Int, nn/2):round(Int,nn/2-1),1,nn)
    kk = convert(Array{eltype(p),2},kk)
    ll = copy(kk)'
    inds = -round(Int,nn/2):round(Int,nn/2-1)
    δ_i_j = zeros(eltype(p),nn,nn)
    for j in 1:nn^2
        # δ_i_j[:,:] .= zeros(eltype(p),nn,nn)
        δ_i_j[:,:] *= zero(eltype(p))
        δ_i_j[j] = one(eltype(p))
        ttjac[:,j] .= (.- 4π^2*(  conv2!(plan,iplan,At,Bt,C,p.*(kk.^2 .+ll.^2),δ_i_j)[inds+nn+1,inds+nn+1]
                                .+conv2!(plan,iplan,At,Bt,C,p.*kk,δ_i_j.*kk)[inds+nn+1,inds+nn+1]
                                .+conv2!(plan,iplan,At,Bt,C,p.*ll,δ_i_j.*ll)[inds+nn+1,inds+nn+1]))[:]
    end
    ttjac
end

function density_ttjac_approx!(ttjac,p,tt,params,filt_matrix)
    plan,iplan,At,Bt,C,vv,f = getindex.(params,[:plan,:iplan,:At,:Bt,:C,:vv,:f])
    nn = size(p)[1]
    nd2 = round(Int,nn/2)
    kk = repmat((-nd2:nd2-1)*(1+0im),1,nn)
    ll = copy(kk)'
    inds = (-nd2:nd2-1)+nn+1

    δ = zeros(eltype(p),nn,nn)
    δ[1] = one(eltype(p))
    v1 = (.- 4π^2*(  conv2!(plan,iplan,At,Bt,C,p.*(kk.^2 .+ll.^2),δ)[inds,inds]
                                .+conv2!(plan,iplan,At,Bt,C,p.*kk,δ.*kk)[inds,inds]
                                .+conv2!(plan,iplan,At,Bt,C,p.*ll,δ.*ll)[inds,inds]))[:]

    δ[1] = zero(eltype(p))
    δ[end] = one(eltype(p))
    v2 = (.- 4π^2*(  conv2!(plan,iplan,At,Bt,C,p.*(kk.^2 .+ll.^2),δ)[inds,inds]
                   .+conv2!(plan,iplan,At,Bt,C,p.*kk,δ.*kk)[inds,inds]
                   .+conv2!(plan,iplan,At,Bt,C,p.*ll,δ.*ll)[inds,inds]))[:]
    v2[1] = v1[1]

    ttjac[:,:] .= (filt_matrix.*ToeplitzMatrices.Toeplitz(v1,v2))[:,:]
end

function temperature!(dtt,tt,p,params)
    plan,iplan,At,Bt,C,vv,A,B,f,T0 = getindex.(params,[:plan,:iplan,:At,:Bt,:C,:vv,:A,:B,:f,:T0])
    nn = size(p)[1]
    nd2 = round(Int,nn/2)
    kk = repmat(-round(Int,nn/2):round(Int,nn/2-1),1,nn)
    ll = copy(kk)'
    delta_k_l_0 = zeros(eltype(p),nn,nn)
    delta_k_l_0[nd2+1,nd2+1] = one(eltype(p))
    inds = (-nd2:nd2-1)+nn+1

    dtt[:, :] .= (  A*(f[1]^2+f[2]^2)*p-2π*im*A*conv2!(plan,iplan,At,Bt,C,(kk*f[1]+ll*f[2]).*p,tt)[inds,inds]
                  - 4π*im*A*conv2!(plan,iplan,At,Bt,C,p,(f[1]*kk+f[2]*ll).*vv)[inds,inds]
                  - 4π^2*A*conv2!(plan,iplan,At,Bt,C,tt,conv2!(plan,iplan,At,Bt,C,p.*kk,vv.*kk)[inds,inds]
                                    +conv2!(plan,iplan,At,Bt,C,p.*ll,vv.*ll)[inds,inds])[inds,inds]
                  - 4π^2*A*conv2!(plan,iplan,At,Bt,C,p,conv2!(plan,iplan,At,Bt,C,vv.*kk,vv.*kk)[inds,inds]
                                   +conv2!(plan,iplan,At,Bt,C,vv.*ll,vv.*ll)[inds,inds])[inds,inds]
                  - 4π^2*B[1]*tt.*(kk.^2 .+ll.^2)
                  .- B[2]*(tt.-T0*delta_k_l_0)
    )
end

function temperature_pjac!(pjac,tt,p,params)
    plan,iplan,At,Bt,C,vv,A,B,f = getindex.(params,[:plan,:iplan,:At,:Bt,:C,:vv,:A,:B,:f])
    nn = size(p)[1]
    nd2 = round(Int,nn/2)
    kk = repmat(-round(Int,nn/2):round(Int,nn/2-1),1,nn)
    ll = copy(kk)'
    delta_k_l_0 = zeros(eltype(p),nn,nn)
    delta_k_l_0[nd2+1,nd2+1] = one(eltype(p))
    inds = (-nd2:nd2-1)+nn+1
    pjac[:,:] .= A*(f[1]^2+f[2]^2)*Diagonal(ones(nn^2))

    δ_i_j = zeros(eltype(p),nn,nn)
    for j in 1:nn^2
        δ_i_j[:,:] .= zeros(eltype(p),nn,nn)
        δ_i_j[j] = one(eltype(p))
        pjac[:,j] .+= ( -2π*im*A*conv2!(plan,iplan,At,Bt,C,(kk*f[1]+ll*f[2]).*δ_i_j,tt)[inds,inds]
                      - 4π*im*A*conv2!(plan,iplan,At,Bt,C,δ_i_j,(f[1]*kk+f[2]*ll).*vv)[inds,inds]
                      - 4π^2*A*conv2!(plan,iplan,At,Bt,C,tt,conv2!(plan,iplan,At,Bt,C,δ_i_j.*kk, vv.*kk)[inds,inds]
                                        +conv2!(plan,iplan,At,Bt,C,δ_i_j.*ll,vv.*ll)[inds,inds])[inds,inds]
                      - 4π^2*A*conv2!(plan,iplan,At,Bt,C,δ_i_j,conv2!(plan,iplan,At,Bt,C,vv.*kk,vv.*kk)[inds,inds]
                                       +conv2!(plan,iplan,At,Bt,C,vv.*ll,vv.*ll)[inds,inds])[inds,inds])[:]
    end
    pjac
end

function temperature_pjac_approx!(pjac,tt,p,params,filt_matrix)
    plan,iplan,At,Bt,C,vv,A,B,f = getindex.(params,[:plan,:iplan,:At,:Bt,:C,:vv,:A,:B,:f])
    nn = size(p)[1]
    nd2 = round(Int,nn/2)
    kk = repmat(-round(Int,nn/2):round(Int,nn/2-1),1,nn)
    ll = copy(kk)'
    delta_k_l_0 = zeros(eltype(p),nn,nn)
    delta_k_l_0[nd2+1,nd2+1] = one(eltype(p))
    inds = (-nd2:nd2-1)+nn+1
    pjac[:,:] .= A*(f[1]^2+f[2]^2)*Diagonal(ones(nn^2))

    δ = zeros(eltype(p),nn,nn)
    δ[1] = one(eltype(p))
    v1 = ( -2π*im*A*conv2!(plan,iplan,At,Bt,C,(kk*f[1].+ll*f[2]).*δ,tt)[inds,inds]
                      .- 4π*im*A*conv2!(plan,iplan,At,Bt,C,δ,(f[1]*kk+f[2]*ll).*vv)[inds,inds]
                      .- 4π^2*A*conv2!(plan,iplan,At,Bt,C,tt,conv2!(plan,iplan,At,Bt,C,δ.*kk, vv.*kk)[inds,inds]
                                        .+conv2!(plan,iplan,At,Bt,C,δ.*ll,vv.*ll)[inds,inds])[inds,inds]
                      .- 4π^2*A*conv2!(plan,iplan,At,Bt,C,δ,conv2!(plan,iplan,At,Bt,C,vv.*kk,vv.*kk)[inds,inds]
                                       .+conv2!(plan,iplan,At,Bt,C,vv.*ll,vv.*ll)[inds,inds])[inds,inds])[:]

    δ[1] *= zero(eltype(p))
    δ[end] = one(eltype(p))
    v2 = ( -2π*im*A*conv2!(plan,iplan,At,Bt,C,(kk*f[1]+ll*f[2]).*δ,tt)[inds,inds]
                      .- 4π*im*A*conv2!(plan,iplan,At,Bt,C,δ,(f[1]*kk+f[2]*ll).*vv)[inds,inds]
                      .- 4π^2*A*conv2!(plan,iplan,At,Bt,C,tt,conv2!(plan,iplan,At,Bt,C,δ.*kk, vv.*kk)[inds,inds]
                                        .+conv2!(plan,iplan,At,Bt,C,δ.*ll,vv.*ll)[inds,inds])[inds,inds]
                      .- 4π^2*A*conv2!(plan,iplan,At,Bt,C,δ,conv2!(plan,iplan,At,Bt,C,vv.*kk,vv.*kk)[inds,inds]
                                       .+conv2!(plan,iplan,At,Bt,C,vv.*ll,vv.*ll)[inds,inds])[inds,inds])[:]
    v2[1] = v1[1]

    pjac[:,:] .+= filt_matrix.*ToeplitzMatrices.Toeplitz(v1,v2)[:,:]
end


function temperature_ttjac!(ttjac,tt,p,params)
    plan,iplan,At,Bt,C,vv,A,B,f = getindex.(params,[:plan,:iplan,:At,:Bt,:C,:vv,:A,:B,:f])
    nn = size(p)[1]
    nd2 = round(Int,nn/2)
    kk = repmat(-round(Int,nn/2):round(Int,nn/2-1),1,nn)
    ll = copy(kk)'
    inds = (-nd2:nd2-1)+nn+1
    delta_k_l_0 = zeros(eltype(p),nn,nn)
    delta_k_l_0[nd2+1,nd2+1] = one(eltype(p))

    ttjac[:,:] .= Diagonal(-4π^2*B[1]*(kk.^2 .+ll.^2)[:]-B[2])
    δ_i_j = zeros(eltype(p),nn,nn)
    for j in 1:nn^2
        δ_i_j[:,:] .= zeros(eltype(p),nn,nn)
        δ_i_j[j] = one(eltype(p))

        ttjac[:,j] .+= ( .-2π*im*A*conv2!(plan,iplan,At,Bt,C,(kk*f[1]+ll*f[2]).*p,δ_i_j)[inds,inds]
                         .-4π^2*A*conv2!(plan,iplan,At,Bt,C,δ_i_j,conv2!(plan,iplan,At,Bt,C,p.*kk,vv.*kk)[inds,inds]
                                           .+conv2!(plan,iplan,At,Bt,C,p.*ll,vv.*ll)[inds,inds])[inds,inds])[:]
    end
    ttjac
end

function temperature_ttjac_approx!(ttjac,tt,p,params,filt_matrix)
    plan,iplan,At,Bt,C,vv,A,B,f = getindex.(params,[:plan,:iplan,:At,:Bt,:C,:vv,:A,:B,:f])
    nn = size(p)[1]
    nd2 = round(Int,nn/2)
    kk = repmat(-round(Int,nn/2):round(Int,nn/2-1),1,nn)
    ll = copy(kk)'
    inds = (-nd2:nd2-1)+nn+1
    delta_k_l_0 = zeros(eltype(p),nn,nn)
    delta_k_l_0[nd2+1,nd2+1] = one(eltype(p))

    ttjac[:,:] .= Diagonal(-4π^2*B[1]*(kk.^2 .+ll.^2)[:]-B[2])

    δ = zeros(eltype(p),nn,nn)
    δ[1] = one(eltype(p))
    v1 = ( .-2π*im*A*conv2!(plan,iplan,At,Bt,C,(kk*f[1].+ll*f[2]).*p,δ)[inds,inds]
           .-4π^2*A*conv2!(plan,iplan,At,Bt,C,δ,conv2!(plan,iplan,At,Bt,C,p.*kk,vv.*kk)[inds,inds]
                           .+conv2!(plan,iplan,At,Bt,C,p.*ll,vv.*ll)[inds,inds])[inds,inds])[:]

    δ[1] = zero(eltype(p))
    δ[end] = one(eltype(p))
    v2 = (.-2π*im*A*conv2!(plan,iplan,At,Bt,C,(kk*f[1].+ll*f[2]).*p,δ)[inds,inds]
           .-4π^2*A*conv2!(plan,iplan,At,Bt,C,δ,conv2!(plan,iplan,At,Bt,C,p.*kk,vv.*kk)[inds,inds]
                           .+conv2!(plan,iplan,At,Bt,C,p.*ll,vv.*ll)[inds,inds])[inds,inds])[:]
    v2[1] = v1[1]

    ttjac[:,:] .+= (filt_matrix.*ToeplitzMatrices.Toeplitz(v1,v2))[:,:]
end

function energy(ϕ, vv, A)
    potential_energy = zero(eltype(ϕ))
    nn = size(vv)[1]
    inds = -round(Int,nn/2):round(Int,nn/2-1)
    tmp = round(Int,nn/2)
    for k in inds
        for l in inds
            for m in inds
                for n in inds
                    (k+m == 0 && n+l == 0) && (potential_energy += ϕ[k+tmp+1,l+tmp+1]*vv[m+tmp+1,n+tmp+1])
                end
            end
        end
    end

    thermal_energy = ϕ[tmp+1, nn+1+tmp]/A

    potential_energy + thermal_energy
end


function real_current(ϕ,params)
    vv = params[:vv]
    nn = size(ϕ)[1]
    nd2 = round(Int,nn/2)
    kk = repmat(-round(Int,nn/2):round(Int,nn/2-1),1,nn)
    ll = copy(kk)'
    p = ϕ[1:nn,1:nn]
    tt = ϕ[1:nn,nn+1:2nn]

    ∇p = -im*2π*[kk.*p,ll.*p]
    ∇tt = -im*2π*[kk.*tt,ll.*tt]
    ∇vv = -im*2π*[kk.*vv,ll.*vv]

    P = fft(fftshift(p))
    T = fft(fftshift(tt))
    ∇P = [fft(fftshift(∇p[1])),fft(fftshift(∇p[2]))]
    ∇T = [fft(fftshift(∇tt[1])),fft(fftshift(∇tt[2]))]
    ∇V = [fft(fftshift(∇vv[1])),fft(fftshift(∇vv[2]))]

    J = -[P.*(∇V[1].-params[:f][1]).+T.*∇P[1],P.*(∇V[2].-params[:f][2]).+T.*∇P[2]]
end
