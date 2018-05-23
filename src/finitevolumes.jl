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
function flux!(du,u,A,coupling,T0,V,Vxshift,Vyshift,V_x,V_y,dx,dy)
    # TODO Make T0 a vector.
    nx,ny = size(V_x)[1],size(V_x)[2]
    P = reshape(view(u,1:nx*ny),nx,ny)
    T = reshape(view(u,(nx*ny+1):2nx*ny),nx,ny)
    Jx = Array{eltype(P)}(nx+1,ny+1)
    Jy = Array{eltype(P)}(nx+1,ny+1)
    T_x = Array{eltype(P)}(nx+1,ny+1)
    T_y = Array{eltype(P)}(nx+1,ny+1)

    for j in 2:ny, i in 2:nx
        Jx[i,j] = -( (P[i-1,j]+P[i,j])*V_x[i,j]
                     +(T[i-1,j]+T[i,j])*(P[i,j]-P[i-1,j])/dx)/2
        T_x[i,j] = (T[i,j]-T[i-1,j])/dx

        Jy[i,j] = -( (P[i,j-1]+P[i,j])*V_y[i,j]
                     +(T[i,j-1]+T[i,j])*(P[i,j]-P[i,j-1])/dy)/2
        T_y[i,j] = (T[i,j]-T[i,j-1])/dy
    end

    for j in 2:ny
        Jx[nx+1,j] = -( (P[nx-1,j]+P[2,j])*V_x[2,j]
                       +(T[nx-1,j]+T[2,j])*(P[2,j]-P[nx-1,j])/dx)/2
        Jx[1,j] = -( (P[nx-1,j]+P[2,j])*V_x[2,j]
                     +(T[nx-1,j]+T[2,j])*(P[2,j]-P[nx-1,j])/dx)/2
        T_x[1,j] = (T[1,j]-T[nx-1,j])/dx
        T_x[nx+1,j] = (T[2,j]-T[nx,j])/dx

        Jy[1,j] = -( (P[1,j-1]+P[1,j])*V_y[1,j]
                     +(T[1,j-1]+T[1,j])*(P[2,j]-P[1,j-1])/dy)/2
        Jy[nx+1,j] = -( (P[2,j-1]+P[2,j])*V_y[2,j]
                        +(T[2,j-1]+T[2,j])*(P[2,j]-P[2,j-1])/dy)/2
        T_y[1,j] = (T[1,j]-T[1,j-1])/dy
        T_y[nx+1,j] = (T[2,j]-T[2,j-1])/dy
    end

    T_x[2:nx,1] .= (T[2:nx,1]-T[1:nx-1,1])/dx
    T_x[nx+1,1] = (T[2,1]-T[1,1])/dx
    T_x[1,1] = (T[1,1]-T[nx-1,1])/dx
    T_x[1:nx+1,ny+1] = zero(eltype(Jy))

    T_y[1:nx,1] .= (T[1:nx,1]-T0)/dy
    T_y[nx+1,1] = (T[2,1]-T0)/dy
    T_y[1:nx,ny+1] .= (T0-T[1:nx,ny])/dy
    T_y[nx+1,ny+1] = (T0-T[2,2])/dy

    Jx[:,1] = zero(eltype(Jy))
    Jx[:,ny+1] = zero(eltype(Jy))
    Jy[:,1] = zero(eltype(Jy))
    Jy[:,ny+1] = zero(eltype(Jy))

    du[1:nx*ny] .= ((Jx[1:nx,1:ny]-Jx[2:nx+1,1:ny])/dy + (Jy[1:nx,1:ny]-Jy[1:nx,2:ny+1])/dx)[:]

    # du[(nx*ny+1):2nx*ny] .= ((Jx[1:ny,1:nx].*Vxshift[1:ny,1:nx]-Jx[1:ny,2:nx+1].*Vxshift[1:ny,2:nx+1]
    #                           -coupling*(T_x[1:ny,1:nx]-T_x[1:ny,2:nx+1]))/dy
    #                          +(Jy[1:ny,1:nx].*Vyshift[1:ny,1:nx]-Jy[2:ny+1,1:nx].*Vyshift[2:ny+1,1:nx]
    #                           -coupling*(T_y[1:ny,1:nx]-T_y[2:ny+1,1:nx]))/dx)[:]
    du[(nx*ny+1):2nx*ny] .= A*(((Jx[1:nx,1:ny].*Vxshift[1:nx,1:ny]-Jx[2:nx+1,1:ny].*Vxshift[2:nx+1,1:ny]
                               -coupling*(T_x[1:nx,1:ny]-T_x[2:nx+1,1:ny]))/dy
                             +(Jy[1:nx,1:ny].*Vyshift[1:nx,1:ny]-Jy[1:nx,2:ny+1].*Vyshift[1:nx,2:ny+1]
                               -coupling*(T_y[1:nx,1:ny]-T_y[1:nx,2:ny+1]))/dx))[:].-A*du[1:nx*ny].*V[:]
    du
end

flux!(du,u,params,t) = flux!(du,u,params...)

function density_currents!(Pmat,Tmat,Jx,Jy,V_x,V_y,dx,dy)
    nx,ny = size(Jx)[1]-1,size(Jx)[2]-1
    for j in 1:ny+1, i in 1:nx+1
        Jx[i,j] = -( (Pmat[i-1,j]+Pmat[i,j])*V_x[i,j]
                    +(Tmat[i-1,j]+Tmat[i,j])*(Pmat[i,j]-Pmat[i-1,j])/dx)/2

        Jy[i,j] = -( (Pmat[i,j-1]+Pmat[i,j])*V_y[i,j]
                    +(Tmat[i,j-1]+Tmat[i,j])*(Pmat[i,j]-Pmat[i,j-1])/dy)/2
    end
    # Impose confining boundary conditions on the currents.
    Jx[:,ny+1] = zero(eltype(Jy))
    Jy[:,1] = zero(eltype(Jy))
    Jy[:,ny+1] = zero(eltype(Jy))

    Jx,Jy
end

function density_flux!(dP,P,Pmat,Tmat,Jx,Jy,V_x,V_y,dx,dy)
    nx,ny = length(indices(Tmat)[1])-2,length(indices(Tmat)[2])-2
    Pmat[1:nx,1:ny] .= reshape(P,nx,ny)
    # Periodicity in the x direction.
    Pmat[0,1:ny] .= Pmat[nx,1:ny]
    Pmat[nx+1,1:ny] .= Pmat[1,1:ny]
    density_currents!(Pmat,Tmat,Jx,Jy,V_x,V_y,dx,dy)
    dP[:] .= ((Jx[1:nx,1:ny].-Jx[2:nx+1,1:ny])/dy.+(Jy[1:nx,1:ny].-Jy[1:nx,2:ny+1])/dx)[:]
end

density_flux!(dP,P,params,t) = density_flux!(dP,P,params...)

# flux[j,i] = -(
#               ( (P[j,i-1]+P[j,i])*V_x[j,i]
#                +(T[j,i-1]+T[j,i])*(P[j,i]-P[j,i-1])/dx)/2
#               -
#               ( (P[j,i]+P[j,i+1])*V_x[j,i+1]
#                +(T[j,i]+T[j,i+1])*(P[j,i+1]-P[j,i])/dx)/2
#               )/dy
#             +( (P[j-1,i]+P[j,i])*V_y[j,i]
#               +(T[j-1,i]+T[j,i])*(P[j,i]-P[j-1,i])/dy)/2
#             -( (P[j,i]+P[j+1,i])*V_y[j+1,i]
#               +(T[j,i]+T[j+1,i])*(P[j+1,i]-P[j,i])/dy)/2 )/dx

# function density_flux!(::Type{Val{:jac}},jac,P,Pmat,Tmat,Jx,Jy,V_x,V_y,dx,dy)
#     ny = length(indices(Tmat)[1])-2
#     nx = length(indices(Tmat)[2])-2
#     diag_0_indices = diagind(jac,0)  # dPij.
#     diag_m1_indices = diagind(jac,-1)  # dPijm1.
#     # diag_mnxm1_indices = diagind(jac,-nx)  # dPim1jm1.
#     diag_mnx_indices = diagind(jac,-nx)  # dPim1j.
#     # diag_mnxp1_indices = diagind(jac,-nx)  # dPim1jp1.
#     diag_p1_indices = diagind(jac,1)  # dPijp1.
#     diag_pnx_indices = diagind(jac,nx)  # dPip1j
#     for row in 1:nx*ny
#         i,j = ind2sub((ny,nx),row)
#         jac[diag_0_indices[row]] = -( (V_x[j,i]-V_x[j,i+1]+(Tmat[j,i-1]-Tmat[j,i+1])/dx)/2/dy
#                                      +(V_y[j,i]-V_y[j+1,i]+(Tmat[j-1,i]-Tmat[j+1,i])/dy)/2/dx)
#     end
#     for row in 2:(nx*ny)
#         i,j = ind2sub((ny,nx),row-1)
#         jac[diag_m1_indices[row-1]] = (V_y[j+1,i]-(Tmat[j,i]+Tmat[j+1,i])/dy)/2/dx
#     end
#     for row in (nx+1):(nx*ny)
#         i,j = ind2sub((ny,nx),row-nx)
#         jac[diag_mnx_indices[row-nx]] = -(V_x[j,i]-(Tmat[j,i-1]+Tmat[j,i])/dx)/2/dy
#     end
#     for row in 1:(nx*ny-1)
#         i,j = ind2sub((ny,nx),row)
#         jac[diag_p1_indices[row]] = -(V_y[j+1,i]+(Tmat[j,i]+Tmat[j+1,i])/dy)/2/dx
#     end
#     for row in 1:(nx*ny-nx-1)
#         i,j = ind2sub((ny,nx),row)
#         jac[diag_pnx_indices[row]] = -(V_x[j,i+1]+(Tmat[j,i]+Tmat[j,i+1])/dx)/2/dy
#     end
#     jac
# end

function density_flux!(::Type{Val{:jac}},jac,P,Pmat,Tmat,Jx,Jy,V_x,V_y,dx,dy)
    ny = length(indices(Tmat)[1])-2
    nx = length(indices(Tmat)[2])-2
    diag_0_indices = diagind(jac,0)  # dPij.
    diag_m1_indices = diagind(jac,-1)  # dPijm1.
    # diag_mnxm1_indices = diagind(jac,-nx)  # dPim1jm1.
    diag_mnx_indices = diagind(jac,-nx)  # dPim1j.
    # diag_mnxp1_indices = diagind(jac,-nx)  # dPim1jp1.
    diag_p1_indices = diagind(jac,1)  # dPijp1.
    diag_pnx_indices = diagind(jac,nx)  # dPip1j
    for row in 1:nx*ny-1
        i,j = ind2sub((nx,ny),row)
        jac[diag_0_indices[row]] = -(Tmat[i-1,j]+Tmat[i,j-1]+4*Tmat[i,j]+Tmat[i,j+1]+Tmat[i+1,j]+dx*(V_x[i,j]-V_x[i+1,j])+dy*(V_y[i,j]-V_y[i,j+1]))/(2*dx*dy)
    end
    for row in 2:(nx*ny)
        i,j = ind2sub((nx,ny),row)
        jac[diag_m1_indices[row-1]] = (Tmat[i-1,j]+Tmat[i,j]-V_x[i,j]*dx)/(2*dx*dy)
    end
    for row in (nx+1):(nx*ny-1)
        i,j = ind2sub((ny,nx),row)
        jac[diag_mnx_indices[row-nx]] = (Tmat[i,j-1]+Tmat[i,j]-V_y[i,j]*dy)/(2*dx*dy)
    end
    for row in 1:(nx*ny-1)
        i,j = ind2sub((nx,ny),row)
        jac[diag_p1_indices[row]] = (Tmat[i,j]+Tmat[i+1,j]+V_x[i+1,j]*dx)/(2*dx*dy)
    end
    for row in 1:(nx*ny-nx-1)
        i,j = ind2sub((nx,ny),row)
        jac[diag_pnx_indices[row]] = (Tmat[i,j]+Tmat[i,j+1]+V_y[i,j+1]*dy)/(2*dx*dy)
    end
    jac
end

density_flux!(::Type{Val{:jac}},jac,P,params,t) = density_flux!(Val{:jac},jac,P,params...)

# function flux_jacobian!(jac,du,u,A,coupling,T0,V,Vxshift,Vyshift,V_x,V_y,dx,dy)
#     # TODO Make T0 a vector.
#     ny,nx = size(V_x)[1],size(V_x)[2]
#     P = reshape(view(u,1:nx*ny),ny,nx)
#     T = reshape(view(u,(nx*ny+1):2nx*ny),ny,nx)
#
#     dP22 = fill(zero(eltype(V_x)),ny,nx)
#     for i in 2:nx, j in 2:ny
#         dP22[j,i] = (V_x[j,i]-V_x[j,i-1])/2dy + (V_x[j,i]-V_y[j-1,i])/2dx
#     end
#     jac[diagind(jac,0)[1:nx*ny]] .= dP22[:]
# end
#
#
# function get_flux(potential, gradpotential_x, gradpotential_y, coupling)
#     @syms _x _y real=true
#     @syms dx dy real=true positive=true
#     @syms P11 P12 P13 P21 P22 P23 P31 P32 P33 real=true positive=true
#     @syms T11 T12 T13 T21 T22 T23 T31 T32 T33 real=true positive=true
#
#     density_flux = (Jx22-Jx12)/dy + (Jy22-Jy21)/dx
#     energy_flux = A*(((Jx22*(V22+V12)/2 - Jx12*Vxshift[1:ny,2:nx+1]
#                                -coupling*(T_x[1:ny,1:nx]-T_x[1:ny,2:nx+1]))/dy
#                              +(Jy[1:ny,1:nx].*Vyshift[1:ny,1:nx]-Jy[2:ny+1,1:nx].*Vyshift[2:ny+1,1:nx]
#                                -coupling*(T_y[1:ny,1:nx]-T_y[2:ny+1,1:nx]))/dx))
#     temperature_flux = energy_flux-A*V22*density_flux
#
#     density_flux, temperature_flux
# end

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
