mutable struct FVMParameters{T}
    Pmat::OffsetArray{T,2,Array{T,2}}
    Tmat::OffsetArray{T,2,Array{T,2}}
    jac_::SparseMatrixCSC{T,Int}
    Jx::Array{T,2}
    Jy::Array{T,2}
    V::Array{T,2}
    Vxshift::Array{T,2}
    Vyshift::Array{T,2}
    V_x::Array{T,2}
    V_y::Array{T,2}
    dx::T
    dy::T
    A::T
    coupling::T
end

function get_variables(mesh,u::AbstractArray{elT,N},params) where elT where N
    xx,yy = mesh
    nx,ny = length(xx),length(yy)
    dx,dy = xx[2]-xx[1],yy[2]-yy[1]
    P = fill(zero(elT),0:ny+1,0:nx+1)
    T = fill(zero(elT),0:ny+1,0:nx+1)

    P[1:ny,1:nx] .= reshape(u[1:nx*ny],ny,nx)
    # Confining boundary conditions in the y direction.
    P[0,0:nx+1] = zero(elT)
    P[ny+1,0:nx+1] = zero(elT)
    # Periodicity in the x direction.
    P[1:ny,0] .= P[1:ny,nx]
    P[1:ny,nx+1] .= P[1:ny,1]
    # Dirichlet boundary conditions in the y direction.
    T0 = params[3]
    T[0,0:nx+1] = T0
    T[ny+1,0:nx+1] = T0
    # Periodicity in the x direction.
    T[1:ny,0] .= T[1:ny,nx]
    T[1:ny,nx+1] .= T[1:ny,1]

    P,T
end

function density_current(mesh,u::AbstractArray{elT,N},params;rotate=false) where elT where N
    @unpack Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling = params
    xx,yy = mesh
    nx = length(xx)
    ny = length(yy)

    Pmat[1:nx,1:ny] .= reshape(u[1:nx*ny],nx,ny)
    # Periodicity in the x direction.
    Pmat[0,1:ny] .= Pmat[nx-1,1:ny]
    Pmat[nx+1,1:ny] .= Pmat[2,1:ny]

    Tmat[1:nx,1:ny] .= reshape(u[(nx*ny+1):end],nx,ny)
    # Periodicity in the x direction.
    Tmat[0,1:ny] .= Tmat[nx-1,1:ny]
    Tmat[nx+1,1:ny] .= Tmat[2,1:ny]
    rotate && (Tmat = rotl90(Tmat); Pmat = rotl90(Pmat); V_x = rotl90(V_x); V_y = rotl90(V_y))

    Jx = [-( 0.5*(Pmat[i-1,j]+Pmat[i,j])*V_x[i,j]
            +0.5*(Tmat[i-1,j]+Tmat[i,j])*(Pmat[i,j]-Pmat[i-1,j])/dx) for i in 1:nx, j in 1:ny]

    Jy = [-( 0.5*(Pmat[i,j-1]+Pmat[i,j])*V_y[i,j]
            +0.5*(Tmat[i,j-1]+Tmat[i,j])*(Pmat[i,j]-Pmat[i,j-1])/dy) for i in 1:nx, j in 1:ny]

    Jx,Jy
end

function flux!(du,u,params)
    @unpack Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling = params
    nx,ny = size(V_x)[1]-1,size(V_x)[2]-1
    T_x = Array{eltype(Pmat)}(undef,nx+1,ny+1)
    T_y = Array{eltype(Pmat)}(undef,nx+1,ny+1)

    Pmat[1:nx,1:ny] .= reshape(u[1:nx*ny],nx,ny)
    # Periodicity in the x direction.
    Pmat[0,1:ny] .= Pmat[nx-1,1:ny]
    Pmat[nx+1,1:ny] .= Pmat[2,1:ny]

    Tmat[1:nx,1:ny] .= reshape(u[(nx*ny+1):end],nx,ny)
    # Periodicity in the x direction.
    Tmat[0,1:ny] .= Tmat[nx-1,1:ny]
    Tmat[nx+1,1:ny] .= Tmat[2,1:ny]
    density_currents!(Pmat,Tmat,Jx,Jy,V_x,V_y,dx,dy)
    temperature_gradients!(T_x,T_y,Tmat,dx,dy)

    du[1:nx*ny] .= ((Jx[1:nx,1:ny].-Jx[2:nx+1,1:ny])/dx.+(Jy[1:nx,1:ny].-Jy[1:nx,2:ny+1])/dy)[:]
    du[(nx*ny+1):2nx*ny] .= A*(((Jx[1:nx,1:ny].*Vxshift[1:nx,1:ny]-Jx[2:nx+1,1:ny].*Vxshift[2:nx+1,1:ny]
                                -coupling*(T_x[1:nx,1:ny].-T_x[2:nx+1,1:ny]))/dx
                               +(Jy[1:nx,1:ny].*Vyshift[1:nx,1:ny]-Jy[1:nx,2:ny+1].*Vyshift[1:nx,2:ny+1]
                                 -coupling*(T_y[1:nx,1:ny].-T_y[1:nx,2:ny+1]))/dy))[:].-A*du[1:nx*ny].*V[:]
    du
end

flux!(du,u,params,t) = flux!(du,u,params)

function flux_autodiff!(du,u,params,t)
    V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling = params

    nx,ny = size(V_x)[1]-1,size(V_x)[2]-1
    T_x = Array{eltype(u)}(nx+1,ny+1)
    T_y = Array{eltype(u)}(nx+1,ny+1)
    Jx = Array{eltype(u)}(nx+1,ny+1)
    Jy = Array{eltype(u)}(nx+1,ny+1)
    Pmat = OffsetArray(eltype(u),0:nx+1,0:ny+1)
    Pmat[0:nx+1,0] = 0
    Pmat[0:nx+1,ny+1] = 0
    Tmat = OffsetArray(eltype(u),0:nx+1,0:ny+1)
    Tmat[0:nx+1,0] = one(eltype(u))  # TODO Somehow pass in T0.
    Tmat[0:nx+1,ny+1] = one(eltype(u))

    Pmat[1:nx,1:ny] .= reshape(u[1:nx*ny],nx,ny)
    # Periodicity in the x direction.
    Pmat[0,1:ny] .= Pmat[nx-1,1:ny]
    Pmat[nx+1,1:ny] .= Pmat[2,1:ny]

    Tmat[1:nx,1:ny] .= reshape(u[(nx*ny+1):end],nx,ny)
    # Periodicity in the x direction.
    Tmat[0,1:ny] .= Tmat[nx-1,1:ny]
    Tmat[nx+1,1:ny] .= Tmat[2,1:ny]
    density_currents!(Pmat,Tmat,Jx,Jy,V_x,V_y,dx,dy)
    temperature_gradients!(T_x,T_y,Tmat,dx,dy)

    du[1:nx*ny] .= ((Jx[1:nx,1:ny].-Jx[2:nx+1,1:ny])/dx.+(Jy[1:nx,1:ny].-Jy[1:nx,2:ny+1])/dy)[:]
    du[(nx*ny+1):2nx*ny] .= A*(((Jx[1:nx,1:ny].*Vxshift[1:nx,1:ny]-Jx[2:nx+1,1:ny].*Vxshift[2:nx+1,1:ny]
                                -coupling*(T_x[1:nx,1:ny].-T_x[2:nx+1,1:ny]))/dx
                               +(Jy[1:nx,1:ny].*Vyshift[1:nx,1:ny]-Jy[1:nx,2:ny+1].*Vyshift[1:nx,2:ny+1]
                                 -coupling*(T_y[1:nx,1:ny].-T_y[1:nx,2:ny+1]))/dy))[:].-A*du[1:nx*ny].*V[:]
    du
end

function flux!(::Type{Val{:jac}},jac,u,params,t)
    nx,ny = size(params.V_x)[1]-1,size(params.V_x)[2]-1

    density_flux!(Val{:jac},view(jac,1:nx*ny,1:nx*ny),u,params,t)
    density_coupling!(Val{:jac},view(jac,1:nx*ny,(nx*ny+1):2*nx*ny),u,params,t)
    temperature_coupling!(Val{:jac},view(jac,(nx*ny+1):2*nx*ny,1:nx*ny),u,params,t)
    temperature_flux!(Val{:jac},view(jac,(nx*ny+1):2*nx*ny,(nx*ny+1):2*nx*ny),u,params,t)
    jac
end

function temperature_gradients!(T_x,T_y,Tmat,dx,dy)
    nx,ny = length(axes(Tmat)[1])-2,length(axes(Tmat)[2])-2
    for j in 1:ny+1, i in 1:nx+1
        T_x[i,j] = (Tmat[i,j]-Tmat[i-1,j])/dx
        T_y[i,j] = (Tmat[i,j]-Tmat[i,j-1])/dy
    end

    T_x,T_y
end

function density_currents!(Pmat,Tmat,Jx,Jy,V_x,V_y,dx,dy)
    nx,ny = size(Jx)[1]-1,size(Jx)[2]-1

    for j in 2:ny, i in 2:nx
        Jx[i,j] = -( (Pmat[i-1,j]+Pmat[i,j])*V_x[i,j]
                    +(Tmat[i-1,j]+Tmat[i,j])*(Pmat[i,j]-Pmat[i-1,j])/dx)/2
        Jy[i,j] = -( (Pmat[i,j-1]+Pmat[i,j])*V_y[i,j]
                    +(Tmat[i,j-1]+Tmat[i,j])*(Pmat[i,j]-Pmat[i,j-1])/dy)/2
    end
    for j in 2:ny
        Jx[nx+1,j] = -( (Pmat[nx,j]+Pmat[2,j])*V_x[2,j]
                       +(Tmat[nx,j]+Tmat[2,j])*(Pmat[2,j]-Pmat[nx,j])/dx)/2
        Jx[1,j] = -( (Pmat[nx-1,j]+Pmat[1,j])*V_x[1,j]
                    +(Tmat[nx-1,j]+Tmat[1,j])*(Pmat[1,j]-Pmat[nx-1,j])/dx)/2

        Jy[1,j] = -( (Pmat[1,j-1]+Pmat[1,j])*V_y[1,j]
                    +(Tmat[1,j-1]+Tmat[1,j])*(Pmat[1,j]-Pmat[1,j-1])/dy)/2
        Jy[nx+1,j] = -( (Pmat[2,j-1]+Pmat[2,j])*V_y[2,j]
                       +(Tmat[2,j-1]+Tmat[2,j])*(Pmat[2,j]-Pmat[2,j-1])/dy)/2
    end

    # Impose confining boundary conditions on the currents.
    Jx[:,1] .= zero(eltype(Jy))
    Jx[:,ny+1] .= zero(eltype(Jy))
    Jy[:,1] .= zero(eltype(Jy))
    Jy[:,ny+1] .= zero(eltype(Jy))

    Jx,Jy
end

function density_flux!(dP,P,Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling)
    nx,ny = length(axes(Tmat)[1])-2,length(axes(Tmat)[2])-2
    Pmat[1:nx,1:ny] .= reshape(P,nx,ny)
    # Periodicity in the x direction.
    Pmat[0,1:ny] .= Pmat[nx-1,1:ny]
    Pmat[nx+1,1:ny] .= Pmat[2,1:ny]
    density_currents!(Pmat,Tmat,Jx,Jy,V_x,V_y,dx,dy)
    dP[:] .= ((Jx[1:nx,1:ny].-Jx[2:nx+1,1:ny])/dy.+(Jy[1:nx,1:ny].-Jy[1:nx,2:ny+1])/dx)[:]
end

density_flux!(dP,P,params,t) = density_flux!(dP,P,params...)

function array_index(row,i,j,nx,ny)
    ((1 <= i <= nx) && (1 <= j <= ny)) && return sub2ind((nx,ny),i,j)
    i == 0 && return sub2ind((nx,ny),nx,mod(j,ny)+1)
    i == nx+1 && return sub2ind((nx,ny),1,mod(j,ny)+1)
    j == 0 && return row
    j == ny+1 && return row
end
