function density_flux!(::Type{Val{:jac}},jac,u,params)
    @unpack Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling = params
    ny = length(indices(Tmat)[1])-2
    nx = length(indices(Tmat)[2])-2
    Pmat[1:nx,1:ny] .= reshape(u[1:nx*ny],nx,ny)
    # Periodicity in the x direction.
    Pmat[0,1:ny] .= Pmat[nx-1,1:ny]
    Pmat[nx+1,1:ny] .= Pmat[2,1:ny]

    Tmat[1:nx,1:ny] .= reshape(u[(nx*ny+1):end],nx,ny)
    # Periodicity in the x direction.
    Tmat[0,1:ny] .= Tmat[nx-1,1:ny]
    Tmat[nx+1,1:ny] .= Tmat[2,1:ny]

    inds = Array{Int64}(5)
    for row in 1:nx*ny
        i,j = ind2sub((nx,ny),row)
        stencil_indices!(inds,(nx,ny),row)
        jac[row,inds] .= [-(Tmat[i-1,j]+Tmat[i,j-1]+4*Tmat[i,j]+Tmat[i,j+1]+Tmat[i+1,j]+dx*(V_x[i,j]-V_x[i+1,j])+dy*(V_y[i,j]-V_y[i,j+1]))/(2*dx*dy),(Tmat[i-1,j]+Tmat[i,j]-V_x[i,j]*dx)/(2*dx*dy),(Tmat[i,j-1]+Tmat[i,j]-V_y[i,j]*dy)/(2*dx*dy),
                          (Tmat[i,j]+Tmat[i+1,j]+V_x[i+1,j]*dx)/(2*dx*dy),(Tmat[i,j]+Tmat[i,j+1]+V_y[i,j+1]*dy)/(2*dx*dy)]
    end
    jac[diagind(jac,nx*ny-1)] = zero(eltype(jac))
    jac[diagind(jac,-nx*ny+1)] = zero(eltype(jac))
    jac[diagind(jac,nx*ny-nx)] = zero(eltype(jac))
    jac[diagind(jac,-nx*ny+nx)] = zero(eltype(jac))
    jac
end

density_flux!(::Type{Val{:jac}},jac,u,params,t) = density_flux!(Val{:jac},jac,u,params)

function density_coupling!(::Type{Val{:jac}},jac,u,params)
    @unpack Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling = params
    ny = length(indices(Tmat)[1])-2
    nx = length(indices(Tmat)[2])-2
    Pmat[1:nx,1:ny] .= reshape(u[1:nx*ny],nx,ny)
    # Periodicity in the x direction.
    Pmat[0,1:ny] .= Pmat[nx-1,1:ny]
    Pmat[nx+1,1:ny] .= Pmat[2,1:ny]

    Tmat[1:nx,1:ny] .= reshape(u[(nx*ny+1):end],nx,ny)
    # Periodicity in the x direction.
    Tmat[0,1:ny] .= Tmat[nx-1,1:ny]
    Tmat[nx+1,1:ny] .= Tmat[2,1:ny]

    inds = Array{Int64}(5)
    for row in 1:nx*ny
        i,j = ind2sub((nx,ny),row)
        stencil_indices!(inds,(nx,ny),row)
        jac[row,inds] .= [(Pmat[i-1,j]+Pmat[i,j-1]-4*Pmat[i,j]+Pmat[i,j+1]+Pmat[i+1,j])/(2*dx*dy),(Pmat[i-1,j]-Pmat[i,j])/(2*dx*dy),(Pmat[i,j-1]-Pmat[i,j])/(2*dx*dy),
                          (-Pmat[i,j]+Pmat[i+1,j])/(2*dx*dy),(-Pmat[i,j]+Pmat[i,j+1])/(2*dx*dy)]
    end
    jac[diagind(jac,nx*ny-1)] = zero(eltype(jac))
    jac[diagind(jac,-nx*ny+1)] = zero(eltype(jac))
    jac[diagind(jac,nx*ny-nx)] = zero(eltype(jac))
    jac[diagind(jac,-nx*ny+nx)] = zero(eltype(jac))
    jac
end

density_coupling!(::Type{Val{:jac}},jac,u,params,t) = density_coupling!(Val{:jac},jac,u,params)

function temperature_coupling!(::Type{Val{:jac}},jac,u,params)
    @unpack Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling = params
    ny = length(indices(Tmat)[1])-2
    nx = length(indices(Tmat)[2])-2
    Pmat[1:nx,1:ny] .= reshape(u[1:nx*ny],nx,ny)
    # Periodicity in the x direction.
    Pmat[0,1:ny] .= Pmat[nx-1,1:ny]
    Pmat[nx+1,1:ny] .= Pmat[2,1:ny]

    Tmat[1:nx,1:ny] .= reshape(u[(nx*ny+1):end],nx,ny)
    # Periodicity in the x direction.
    Tmat[0,1:ny] .= Tmat[nx-1,1:ny]
    Tmat[nx+1,1:ny] .= Tmat[2,1:ny]

    inds = Array{Int64}(5)
    for row in 1:nx*ny
        i,j = ind2sub((nx,ny),row)
        stencil_indices!(inds,(nx,ny),row)
        jac[row,inds] .= [-A*(-V[i,j]*(Tmat[i-1,j]+Tmat[i,j-1]+4*Tmat[i,j]+Tmat[i,j+1]+Tmat[i+1,j]+dx*(V_x[i,j]-V_x[i+1,j])+dy*(V_y[i,j]-V_y[i,j+1]))+Vxshift[i,j]*(Tmat[i-1,j]+Tmat[i,j]+V_x[i,j]*dx)+Vxshift[i+1,j]*(Tmat[i,j]+Tmat[i+1,j]-V_x[i+1,j]*dx)+Vyshift[i,j]*(Tmat[i,j-1]+Tmat[i,j]+V_y[i,j]*dy)+Vyshift[i,j+1]*(Tmat[i,j]+Tmat[i,j+1]-V_y[i,j+1]*dy))/(2*dx*dy),-A*(V[i,j]-Vxshift[i,j])*(Tmat[i-1,j]+Tmat[i,j]-V_x[i,j]*dx)/(2*dx*dy),-A*(V[i,j]-Vyshift[i,j])*(Tmat[i,j-1]+Tmat[i,j]-V_y[i,j]*dy)/(2*dx*dy),
                          -A*(V[i,j]-Vxshift[i+1,j])*(Tmat[i,j]+Tmat[i+1,j]+V_x[i+1,j]*dx)/(2*dx*dy),-A*(V[i,j]-Vyshift[i,j+1])*(Tmat[i,j]+Tmat[i,j+1]+V_y[i,j+1]*dy)/(2*dx*dy)]
    end
    jac[diagind(jac,nx*ny-1)] = zero(eltype(jac))
    jac[diagind(jac,-nx*ny+1)] = zero(eltype(jac))
    jac[diagind(jac,nx*ny-nx)] = zero(eltype(jac))
    jac[diagind(jac,-nx*ny+nx)] = zero(eltype(jac))
    jac
end

temperature_coupling!(::Type{Val{:jac}},jac,u,params,t) = temperature_coupling!(Val{:jac},jac,u,params)

function temperature_flux!(::Type{Val{:jac}},jac,u,params)
    @unpack Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling = params
    ny = length(indices(Tmat)[1])-2
    nx = length(indices(Tmat)[2])-2
    Pmat[1:nx,1:ny] .= reshape(u[1:nx*ny],nx,ny)
    # Periodicity in the x direction.
    Pmat[0,1:ny] .= Pmat[nx-1,1:ny]
    Pmat[nx+1,1:ny] .= Pmat[2,1:ny]

    Tmat[1:nx,1:ny] .= reshape(u[(nx*ny+1):end],nx,ny)
    # Periodicity in the x direction.
    Tmat[0,1:ny] .= Tmat[nx-1,1:ny]
    Tmat[nx+1,1:ny] .= Tmat[2,1:ny]

    inds = Array{Int64}(5)
    for row in 1:nx*ny
        i,j = ind2sub((nx,ny),row)
        stencil_indices!(inds,(nx,ny),row)
        jac[row,inds] .= [-A*(8*coupling+V[i,j]*(Pmat[i-1,j]+Pmat[i,j-1]-4*Pmat[i,j]+Pmat[i,j+1]+Pmat[i+1,j])-Vxshift[i,j]*(Pmat[i-1,j]-Pmat[i,j])+Vxshift[i+1,j]*(Pmat[i,j]-Pmat[i+1,j])-Vyshift[i,j]*(Pmat[i,j-1]-Pmat[i,j])+Vyshift[i,j+1]*(Pmat[i,j]-Pmat[i,j+1]))/(2*dx*dy),A*(2*coupling-V[i,j]*(Pmat[i-1,j]-Pmat[i,j])+Vxshift[i,j]*(Pmat[i-1,j]-Pmat[i,j]))/(2*dx*dy),A*(2*coupling-V[i,j]*(Pmat[i,j-1]-Pmat[i,j])+Vyshift[i,j]*(Pmat[i,j-1]-Pmat[i,j]))/(2*dx*dy),
                          A*(2*coupling+V[i,j]*(Pmat[i,j]-Pmat[i+1,j])-Vxshift[i+1,j]*(Pmat[i,j]-Pmat[i+1,j]))/(2*dx*dy),A*(2*coupling+V[i,j]*(Pmat[i,j]-Pmat[i,j+1])-Vyshift[i,j+1]*(Pmat[i,j]-Pmat[i,j+1]))/(2*dx*dy)]
    end
    jac[diagind(jac,nx*ny-1)] = zero(eltype(jac))
    jac[diagind(jac,-nx*ny+1)] = zero(eltype(jac))
    jac[diagind(jac,nx*ny-nx)] = zero(eltype(jac))
    jac[diagind(jac,-nx*ny+nx)] = zero(eltype(jac))
    jac
end

temperature_flux!(::Type{Val{:jac}},jac,u,params,t) = temperature_flux!(Val{:jac},jac,u,params)
