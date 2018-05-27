function get_variables(mesh,u::AbstractArray{elT,N},params) where elT where N
    xx,yy = mesh
    nx,ny = length(xx),length(yy)
    dx,dy = (xx[end]-xx[1])/nx,(yy[end]-yy[1])/ny
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

function density_current(mesh,u::AbstractArray{elT,N},params) where elT where N
    A,coupling,T0,V,Vxshift,Vyshift,V_x,V_y,dx,dy = params
    xx,yy = mesh
    nx = length(xx)
    ny = length(yy)

    P,T = get_variables(mesh,u,params)

    Jx = Array{elT}(ny,nx)
    Jy = Array{elT}(ny,nx)
    for i in 1:nx, j in 1:ny
        Jx[j,i] = -( (P[j,i-1]+P[j,i])*V_x[j,i]
                     +(T[j,i-1]+T[j,i])*(P[j,i]-P[j,i-1])/dx)/2
        Jy[j,i] = -( (P[j-1,i]+P[j,i])*V_y[j,i]
                     +(T[j-1,i]+T[j,i])*(P[j,i]-P[j-1,i])/dy)/2
    end

    Jx,Jy
end

"""
   flux!(du,u,A,coupling,T0,V,Vxshift,Vyshift,V_x,V_y,dx,dy)

Compute the flux `du` going out of each cell in `u`, where `u` represents `[P[:];U[:]]`,
where `P[j,i]` is the integrated probability distribution of the `i`, `j`th. `U` is the
energy density. The parameters are as follows:
  * A: Dimensionless inverse heat capacity
  * coupling: Coupling constant
  * T0: Boundary temperature
  * V[j,i]: Potential at the point (x_i,y_j)
  * Vxshift: Potential at the point (x_i-dx,y_j)
  * Vyshift: Potential at the point (x_i,y_j-dy)
  ...
"""
# function flux!(du,u,A,coupling,T0,V,Vxshift,Vyshift,V_x,V_y,dx,dy)
#     # TODO Make T0 a vector.
#     nx,ny = size(V_x)[1],size(V_x)[2]
#     P = reshape(view(u,1:nx*ny),nx,ny)
#     T = reshape(view(u,(nx*ny+1):2nx*ny),nx,ny)
#     Jx = Array{eltype(P)}(nx+1,ny+1)
#     Jy = Array{eltype(P)}(nx+1,ny+1)
#     T_x = Array{eltype(P)}(nx+1,ny+1)
#     T_y = Array{eltype(P)}(nx+1,ny+1)
#
#     for j in 2:ny, i in 2:nx
#         Jx[i,j] = -( (P[i-1,j]+P[i,j])*V_x[i,j]
#                      +(T[i-1,j]+T[i,j])*(P[i,j]-P[i-1,j])/dx)/2
#         T_x[i,j] = (T[i,j]-T[i-1,j])/dx
#
#         Jy[i,j] = -( (P[i,j-1]+P[i,j])*V_y[i,j]
#                      +(T[i,j-1]+T[i,j])*(P[i,j]-P[i,j-1])/dy)/2
#         T_y[i,j] = (T[i,j]-T[i,j-1])/dy
#     end
#
#     for j in 2:ny
#         Jx[nx+1,j] = -( (P[nx-1,j]+P[2,j])*V_x[2,j]
#                        +(T[nx-1,j]+T[2,j])*(P[2,j]-P[nx-1,j])/dx)/2
#         Jx[1,j] = -( (P[nx-1,j]+P[2,j])*V_x[2,j]
#                      +(T[nx-1,j]+T[2,j])*(P[2,j]-P[nx-1,j])/dx)/2
#         T_x[1,j] = (T[1,j]-T[nx-1,j])/dx
#         T_x[nx+1,j] = (T[2,j]-T[nx,j])/dx
#
#         Jy[1,j] = -( (P[1,j-1]+P[1,j])*V_y[1,j]
#                      +(T[1,j-1]+T[1,j])*(P[2,j]-P[1,j-1])/dy)/2
#         Jy[nx+1,j] = -( (P[2,j-1]+P[2,j])*V_y[2,j]
#                         +(T[2,j-1]+T[2,j])*(P[2,j]-P[2,j-1])/dy)/2
#         T_y[1,j] = (T[1,j]-T[1,j-1])/dy
#         T_y[nx+1,j] = (T[2,j]-T[2,j-1])/dy
#     end
#
#     T_x[2:nx,1] .= (T[2:nx,1]-T[1:nx-1,1])/dx
#     T_x[nx+1,1] = (T[2,1]-T[1,1])/dx
#     T_x[1,1] = (T[1,1]-T[nx-1,1])/dx
#     T_x[1:nx+1,ny+1] = zero(eltype(Jy))
#
#     T_y[1:nx,1] .= (T[1:nx,1]-T0)/dy
#     T_y[nx+1,1] = (T[2,1]-T0)/dy
#     T_y[1:nx,ny+1] .= (T0-T[1:nx,ny])/dy
#     T_y[nx+1,ny+1] = (T0-T[2,2])/dy
#
#     Jx[:,1] = zero(eltype(Jy))
#     Jx[:,ny+1] = zero(eltype(Jy))
#     Jy[:,1] = zero(eltype(Jy))
#     Jy[:,ny+1] = zero(eltype(Jy))
#
#     du[1:nx*ny] .= ((Jx[1:nx,1:ny].-Jx[2:nx+1,1:ny])/dy.+(Jy[1:nx,1:ny].-Jy[1:nx,2:ny+1])/dx)[:]
#
#     # du[(nx*ny+1):2nx*ny] .= ((Jx[1:ny,1:nx].*Vxshift[1:ny,1:nx]-Jx[1:ny,2:nx+1].*Vxshift[1:ny,2:nx+1]
#     #                           -coupling*(T_x[1:ny,1:nx]-T_x[1:ny,2:nx+1]))/dy
#     #                          +(Jy[1:ny,1:nx].*Vyshift[1:ny,1:nx]-Jy[2:ny+1,1:nx].*Vyshift[2:ny+1,1:nx]
#     #                           -coupling*(T_y[1:ny,1:nx]-T_y[2:ny+1,1:nx]))/dx)[:]
#     du[(nx*ny+1):2nx*ny] .= A*(((Jx[1:nx,1:ny].*Vxshift[1:nx,1:ny]-Jx[2:nx+1,1:ny].*Vxshift[2:nx+1,1:ny]
#                                -coupling*(T_x[1:nx,1:ny]-T_x[2:nx+1,1:ny]))/dy
#                              +(Jy[1:nx,1:ny].*Vyshift[1:nx,1:ny]-Jy[1:nx,2:ny+1].*Vyshift[1:nx,2:ny+1]
#                                -coupling*(T_y[1:nx,1:ny]-T_y[1:nx,2:ny+1]))/dx))[:].-A*du[1:nx*ny].*V[:]
#     du
# end

function flux!(du,u,Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling)
    nx,ny = size(V_x)[1]-1,size(V_x)[2]-1
    T_x = Array{eltype(Pmat)}(nx+1,ny+1)
    T_y = Array{eltype(Pmat)}(nx+1,ny+1)

    Pmat[1:nx,1:ny] .= reshape(u[1:nx*ny],nx,ny)
    # Periodicity in the x direction.
    Pmat[0,1:ny] .= Pmat[nx,1:ny]
    Pmat[nx+1,1:ny] .= Pmat[1,1:ny]

    Tmat[1:nx,1:ny] .= reshape(u[(nx*ny+1):end],nx,ny)
    # Periodicity in the x direction.
    Tmat[0,1:ny] .= Tmat[nx,1:ny]
    Tmat[nx+1,1:ny] .= Tmat[1,1:ny]
    density_currents!(Pmat,Tmat,Jx,Jy,V_x,V_y,dx,dy)
    temperature_gradients!(T_x,T_y,Tmat,dx,dy)

    du[1:nx*ny] .= ((Jx[1:nx,1:ny].-Jx[2:nx+1,1:ny])/dy.+(Jy[1:nx,1:ny].-Jy[1:nx,2:ny+1])/dx)[:]
    du[(nx*ny+1):2nx*ny] .= A*(((Jx[1:nx,1:ny].*Vxshift[1:nx,1:ny]-Jx[2:nx+1,1:ny].*Vxshift[2:nx+1,1:ny]
                                -coupling*(T_x[1:nx,1:ny]-T_x[2:nx+1,1:ny]))/dy
                               +(Jy[1:nx,1:ny].*Vyshift[1:nx,1:ny]-Jy[1:nx,2:ny+1].*Vyshift[1:nx,2:ny+1]
                                 -coupling*(T_y[1:nx,1:ny]-T_y[1:nx,2:ny+1]))/dx))[:].-A*du[1:nx*ny].*V[:]
    du
end

flux!(du,u,params,t) = flux!(du,u,params...)

function flux!(::Type{Val{:jac}},jac,u,params,t)
    Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling  = params
    nx,ny = size(V_x)[1]-1,size(V_x)[2]-1
    jac[1:nx*ny,1:nx*ny] .= density_flux!(Val{:jac},jac[1:nx*ny,1:nx*ny],u,params,t)
    jac[1:nx*ny,(nx*ny+1):end] .= density_coupling!(Val{:jac},jac[1:nx*ny,1:nx*ny],u,params,t)
    jac[(nx*ny+1):end,1:nx*ny] .= temperature_coupling!(Val{:jac},jac[1:nx*ny,1:nx*ny],u,params,t)
    jac[(nx*ny+1):end,(nx*ny+1):end] .= temperature_flux!(Val{:jac},jac[1:nx*ny,1:nx*ny],u,params,t)
    jac
end

function temperature_gradients!(T_x,T_y,Tmat,dx,dy)
    nx,ny = length(indices(Tmat)[1])-2,length(indices(Tmat)[2])-2
    for j in 1:ny+1, i in 1:nx+1
        T_x[i,j] = (Tmat[i,j]-Tmat[i-1,j])/dx
        T_y[i,j] = (Tmat[i,j]-Tmat[i,j-1])/dy
    end
    T_x,T_y
end

function density_currents!(Pmat,Tmat,Jx,Jy,V_x,V_y,dx,dy)
    nx,ny = size(Jx)[1]-1,size(Jx)[2]-1
    # for j in 1:ny+1, i in 1:nx+1
    #     Jx[i,j] = -( (Pmat[i-1,j]+Pmat[i,j])*V_x[i,j]
    #                 +(Tmat[i-1,j]+Tmat[i,j])*(Pmat[i,j]-Pmat[i-1,j])/dx)/2
    #     Jy[i,j] = -( (Pmat[i,j-1]+Pmat[i,j])*V_y[i,j]
    #                 +(Tmat[i,j-1]+Tmat[i,j])*(Pmat[i,j]-Pmat[i,j-1])/dy)/2
    # end
    for j in 2:ny, i in 2:nx
        Jx[i,j] = -( (Pmat[i-1,j]+Pmat[i,j])*V_x[i,j]
                     +(Tmat[i-1,j]+Tmat[i,j])*(Pmat[i,j]-Pmat[i-1,j])/dx)/2
        Jy[i,j] = -( (Pmat[i,j-1]+Pmat[i,j])*V_y[i,j]
                     +(Tmat[i,j-1]+Tmat[i,j])*(Pmat[i,j]-Pmat[i,j-1])/dy)/2
    end
    for j in 2:ny
        Jx[nx+1,j] = -( (Pmat[nx-1,j]+Pmat[2,j])*V_x[2,j]
                       +(Tmat[nx-1,j]+Tmat[2,j])*(Pmat[2,j]-Pmat[nx-1,j])/dx)/2
        Jx[1,j] = -( (Pmat[nx-1,j]+Pmat[2,j])*V_x[2,j]
                     +(Tmat[nx-1,j]+Tmat[2,j])*(Pmat[2,j]-Pmat[nx-1,j])/dx)/2

        Jy[1,j] = -( (Pmat[1,j-1]+Pmat[1,j])*V_y[1,j]
                     +(Tmat[1,j-1]+Tmat[1,j])*(Pmat[2,j]-Pmat[1,j-1])/dy)/2
        Jy[nx+1,j] = -( (Pmat[2,j-1]+Pmat[2,j])*V_y[2,j]
                        +(Tmat[2,j-1]+Tmat[2,j])*(Pmat[2,j]-Pmat[2,j-1])/dy)/2
    end

    # Impose confining boundary conditions on the currents.
    Jx[:,1] = zero(eltype(Jy))
    Jx[:,ny+1] = zero(eltype(Jy))
    Jy[:,1] = zero(eltype(Jy))
    Jy[:,ny+1] = zero(eltype(Jy))
    # Jx[:,ny+1] = zero(eltype(Jy))
    # Jy[:,1] = zero(eltype(Jy))
    # Jy[:,ny+1] = zero(eltype(Jy))

    Jx,Jy
end

function density_flux!(dP,P,Pmat,Tmat,Jx,Jy,V_x,V_y,dx,dy)
    nx,ny = length(indices(Tmat)[1])-2,length(indices(Tmat)[2])-2
    Pmat[1:nx,1:ny] .= reshape(P,nx,ny)
    # Periodicity in the x direction.
    Pmat[0,1:ny] .= Pmat[nx-1,1:ny]
    Pmat[nx+1,1:ny] .= Pmat[2,1:ny]
    density_currents!(Pmat,Tmat,Jx,Jy,V_x,V_y,dx,dy)
    dP[:] .= ((Jx[1:nx,1:ny].-Jx[2:nx+1,1:ny])/dy.+(Jy[1:nx,1:ny].-Jy[1:nx,2:ny+1])/dx)[:]
end

density_flux!(dP,P,params,t) = density_flux!(dP,P,params...)

# The following code is generated by running the file src/symbolics.jl.
function density_flux!(::Type{Val{:jac}},jac,u,Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling)
    ny = length(indices(Tmat)[1])-2
    nx = length(indices(Tmat)[2])-2
    diag_0_indices = diagind(jac,0)  # dPij.
    diag_m1_indices = [diagind(jac,nx*ny-1);diagind(jac,-1)]  # dPijm1.
    # diag_mnxm1_indices = diagind(jac,-nx)  # dPim1jm1.
    diag_mnx_indices = [diagind(jac,nx*ny-nx);diagind(jac,-nx)]  # dPim1j.
    # diag_mnxp1_indices = diagind(jac,-nx)  # dPim1jp1.
    diag_p1_indices = [diagind(jac,1);diagind(jac,-nx*ny+1)]  # dPijp1.
    diag_pnx_indices = [diagind(jac,nx);diagind(jac,-nx*ny+nx)]  # dPip1j
    diag_p1_indices = [diagind(jac,1);diagind(jac,-nx*ny+1)]  # dPijp1.
    diag_pnx_indices = [diagind(jac,nx);diagind(jac,-nx*ny+nx)]  # dPip1j
    for row in 1:nx*ny
        i,j = ind2sub((nx,ny),row)
        jac[diag_0_indices[row]] = -(Tmat[i-1,j]+Tmat[i,j-1]+4*Tmat[i,j]+Tmat[i,j+1]+Tmat[i+1,j]+dx*(V_x[i,j]-V_x[i+1,j])+dy*(V_y[i,j]-V_y[i,j+1]))/(2*dx*dy)
        jac[diag_m1_indices[row]] = (Tmat[i-1,j]+Tmat[i,j]-V_x[i,j]*dx)/(2*dx*dy)
        jac[diag_mnx_indices[row]] = (Tmat[i,j-1]+Tmat[i,j]-V_y[i,j]*dy)/(2*dx*dy)
        jac[diag_p1_indices[row]] = (Tmat[i,j]+Tmat[i+1,j]+V_x[i+1,j]*dx)/(2*dx*dy)
        jac[diag_pnx_indices[row]] = (Tmat[i,j]+Tmat[i,j+1]+V_y[i,j+1]*dy)/(2*dx*dy)
    end
    jac[diagind(jac,nx-1)[1:nx:end-nx]] .= jac[diagind(jac,-1)[nx:nx:end]]
    jac[diagind(jac,-nx+1)[1:nx:end-nx]] .= jac[diagind(jac,1)[nx:nx:end]]
    jac[diagind(jac,-1)[nx:nx:end]] = 0
    jac[diagind(jac,1)[nx:nx:end]] = 0
    jac[diagind(jac,nx*ny-1)] = 0
    jac[diagind(jac,-nx*ny+1)] = 0
    jac[diagind(jac,nx*ny-nx)] = 0
    jac[diagind(jac,-nx*ny+nx)] = 0
    jac
end

density_flux!(::Type{Val{:jac}},jac,P,params,t) = density_flux!(Val{:jac},jac,P,params...)

function density_coupling!(::Type{Val{:jac}},jac,u,Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling)
    ny = length(indices(Tmat)[1])-2
    nx = length(indices(Tmat)[2])-2
    diag_0_indices = diagind(jac,0)  # dPij.
    diag_m1_indices = [diagind(jac,nx*ny-1);diagind(jac,-1)]  # dPijm1.
    # diag_mnxm1_indices = diagind(jac,-nx)  # dPim1jm1.
    diag_mnx_indices = [diagind(jac,nx*ny-nx);diagind(jac,-nx)]  # dPim1j.
    # diag_mnxp1_indices = diagind(jac,-nx)  # dPim1jp1.
    diag_p1_indices = [diagind(jac,1);diagind(jac,-nx*ny+1)]  # dPijp1.
    diag_pnx_indices = [diagind(jac,nx);diagind(jac,-nx*ny+nx)]  # dPip1j
    diag_p1_indices = [diagind(jac,1);diagind(jac,-nx*ny+1)]  # dPijp1.
    diag_pnx_indices = [diagind(jac,nx);diagind(jac,-nx*ny+nx)]  # dPip1j
    for row in 1:nx*ny
        i,j = ind2sub((nx,ny),row)
        jac[diag_0_indices[row]] = (Pmat[i-1,j]+Pmat[i,j-1]-4*Pmat[i,j]+Pmat[i,j+1]+Pmat[i+1,j])/(2*dx*dy)
        jac[diag_m1_indices[row]] = (Pmat[i-1,j]-Pmat[i,j])/(2*dx*dy)
        jac[diag_mnx_indices[row]] = (Pmat[i,j-1]-Pmat[i,j])/(2*dx*dy)
        jac[diag_p1_indices[row]] = (-Pmat[i,j]+Pmat[i+1,j])/(2*dx*dy)
        jac[diag_pnx_indices[row]] = (-Pmat[i,j]+Pmat[i,j+1])/(2*dx*dy)
    end
    jac[diagind(jac,nx-1)[1:nx:end-nx]] .= jac[diagind(jac,-1)[nx:nx:end]]
    jac[diagind(jac,-nx+1)[1:nx:end-nx]] .= jac[diagind(jac,1)[nx:nx:end]]
    jac[diagind(jac,-1)[nx:nx:end]] = 0
    jac[diagind(jac,1)[nx:nx:end]] = 0
    jac[diagind(jac,nx*ny-1)] = 0
    jac[diagind(jac,-nx*ny+1)] = 0
    jac[diagind(jac,nx*ny-nx)] = 0
    jac[diagind(jac,-nx*ny+nx)] = 0
    jac
end

density_coupling!(::Type{Val{:jac}},jac,P,params,t) = density_coupling!(Val{:jac},jac,P,params...)

function temperature_coupling!(::Type{Val{:jac}},jac,u,Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling)
    ny = length(indices(Tmat)[1])-2
    nx = length(indices(Tmat)[2])-2
    diag_0_indices = diagind(jac,0)  # dPij.
    diag_m1_indices = [diagind(jac,nx*ny-1);diagind(jac,-1)]  # dPijm1.
    # diag_mnxm1_indices = diagind(jac,-nx)  # dPim1jm1.
    diag_mnx_indices = [diagind(jac,nx*ny-nx);diagind(jac,-nx)]  # dPim1j.
    # diag_mnxp1_indices = diagind(jac,-nx)  # dPim1jp1.
    diag_p1_indices = [diagind(jac,1);diagind(jac,-nx*ny+1)]  # dPijp1.
    diag_pnx_indices = [diagind(jac,nx);diagind(jac,-nx*ny+nx)]  # dPip1j
    diag_p1_indices = [diagind(jac,1);diagind(jac,-nx*ny+1)]  # dPijp1.
    diag_pnx_indices = [diagind(jac,nx);diagind(jac,-nx*ny+nx)]  # dPip1j
    for row in 1:nx*ny
        i,j = ind2sub((nx,ny),row)
        jac[diag_0_indices[row]] = A*(Tmat[i-1,j]+Tmat[i,j-1]+4*Tmat[i,j]+Tmat[i,j+1]+Tmat[i+1,j]-Vxshift[i,j]*(Tmat[i-1,j]+Tmat[i,j]+V_x[i,j]*dx)-Vxshift[i+1,j]*(Tmat[i,j]+Tmat[i+1,j]-V_x[i+1,j]*dx)-Vyshift[i,j]*(Tmat[i,j-1]+Tmat[i,j]+V_y[i,j]*dy)+dx*(V_x[i,j]-V_x[i+1,j])+dy*(V_y[i,j]-V_y[i,j+1]))/(2*dx*dy)
        jac[diag_m1_indices[row]] = -A*(Tmat[i-1,j]+Tmat[i,j]-V_x[i,j]*dx-Vxshift[i,j]*(Tmat[i-1,j]+Tmat[i,j]-V_x[i,j]*dx))/(2*dx*dy)
        jac[diag_mnx_indices[row]] = -A*(Tmat[i,j-1]+Tmat[i,j]-V_y[i,j]*dy-Vyshift[i,j]*(Tmat[i,j-1]+Tmat[i,j]-V_y[i,j]*dy))/(2*dx*dy)
        jac[diag_p1_indices[row]] = -A*(Tmat[i,j]+Tmat[i+1,j]+V_x[i+1,j]*dx-Vxshift[i+1,j]*(Tmat[i,j]+Tmat[i+1,j]+V_x[i+1,j]*dx))/(2*dx*dy)
        jac[diag_pnx_indices[row]] = -A*(Tmat[i,j]+Tmat[i,j+1]+V_y[i,j+1]*dy)/(2*dx*dy)
    end
    jac[diagind(jac,nx-1)[1:nx:end-nx]] .= jac[diagind(jac,-1)[nx:nx:end]]
    jac[diagind(jac,-nx+1)[1:nx:end-nx]] .= jac[diagind(jac,1)[nx:nx:end]]
    jac[diagind(jac,-1)[nx:nx:end]] = 0
    jac[diagind(jac,1)[nx:nx:end]] = 0
    jac[diagind(jac,nx*ny-1)] = 0
    jac[diagind(jac,-nx*ny+1)] = 0
    jac[diagind(jac,nx*ny-nx)] = 0
    jac[diagind(jac,-nx*ny+nx)] = 0
    jac
end

temperature_coupling!(::Type{Val{:jac}},jac,P,params,t) = temperature_coupling!(Val{:jac},jac,P,params...)

function temperature_flux!(::Type{Val{:jac}},jac,u,Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling)
    ny = length(indices(Tmat)[1])-2
    nx = length(indices(Tmat)[2])-2
    diag_0_indices = diagind(jac,0)  # dPij.
    diag_m1_indices = [diagind(jac,nx*ny-1);diagind(jac,-1)]  # dPijm1.
    # diag_mnxm1_indices = diagind(jac,-nx)  # dPim1jm1.
    diag_mnx_indices = [diagind(jac,nx*ny-nx);diagind(jac,-nx)]  # dPim1j.
    # diag_mnxp1_indices = diagind(jac,-nx)  # dPim1jp1.
    diag_p1_indices = [diagind(jac,1);diagind(jac,-nx*ny+1)]  # dPijp1.
    diag_pnx_indices = [diagind(jac,nx);diagind(jac,-nx*ny+nx)]  # dPip1j
    diag_p1_indices = [diagind(jac,1);diagind(jac,-nx*ny+1)]  # dPijp1.
    diag_pnx_indices = [diagind(jac,nx);diagind(jac,-nx*ny+nx)]  # dPip1j
    for row in 1:nx*ny
        i,j = ind2sub((nx,ny),row)
        jac[diag_0_indices[row]] = A*(4*coupling*dy+Vyshift[i,j]*dx*(Pmat[i,j-1]-Pmat[i,j])-dx*(-4*coupling+Pmat[i-1,j]+Pmat[i,j-1]-4*Pmat[i,j]+Pmat[i,j+1]+Pmat[i+1,j]-Vxshift[i,j]*(Pmat[i-1,j]-Pmat[i,j])+Vxshift[i+1,j]*(Pmat[i,j]-Pmat[i+1,j])))/(2*dx^2*dy)
        jac[diag_m1_indices[row]] = -A*(2*coupling*dy+dx*(2*coupling+Pmat[i-1,j]-Pmat[i,j]-Vxshift[i,j]*(Pmat[i-1,j]-Pmat[i,j])))/(2*dx^2*dy)
        jac[diag_mnx_indices[row]] = A*(-Pmat[i,j-1]+Pmat[i,j]+Vyshift[i,j]*(Pmat[i,j-1]-Pmat[i,j]))/(2*dx*dy)
        jac[diag_p1_indices[row]] = -A*(-Pmat[i,j]+Pmat[i+1,j]+Vxshift[i+1,j]*(Pmat[i,j]-Pmat[i+1,j]))/(2*dx*dy)
        jac[diag_pnx_indices[row]] = -A*(2*coupling*dy+dx*(2*coupling-Pmat[i,j]+Pmat[i,j+1]))/(2*dx^2*dy)
    end
    jac[diagind(jac,nx-1)[1:nx:end-nx]] .= jac[diagind(jac,-1)[nx:nx:end]]
    jac[diagind(jac,-nx+1)[1:nx:end-nx]] .= jac[diagind(jac,1)[nx:nx:end]]
    jac[diagind(jac,-1)[nx:nx:end]] = 0
    jac[diagind(jac,1)[nx:nx:end]] = 0
    jac[diagind(jac,nx*ny-1)] = 0
    jac[diagind(jac,-nx*ny+1)] = 0
    jac[diagind(jac,nx*ny-nx)] = 0
    jac[diagind(jac,-nx*ny+nx)] = 0
    jac
end

temperature_flux!(::Type{Val{:jac}},jac,P,params,t) = temperature_flux!(Val{:jac},jac,P,params...)
