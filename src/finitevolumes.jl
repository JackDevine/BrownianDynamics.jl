# TODO replace J_x with Jx

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
    ny,nx = size(V_x)[1],size(V_x)[2]
    P = reshape(view(u,1:nx*ny),ny,nx)
    T = reshape(view(u,(nx*ny+1):2nx*ny),ny,nx)
    J_x = Array{eltype(P)}(ny+1,nx+1)
    J_y = Array{eltype(P)}(ny+1,nx+1)
    T_x = Array{eltype(P)}(ny+1,nx+1)
    T_y = Array{eltype(P)}(ny+1,nx+1)

    for i in 2:nx, j in 2:ny
        J_x[j,i] = -( (P[j,i-1]+P[j,i])*V_x[j,i]
                     +(T[j,i-1]+T[j,i])*(P[j,i]-P[j,i-1])/dx)/2
        T_x[j,i] = (T[j,i]-T[j,i-1])/dx

        J_y[j,i] = -( (P[j-1,i]+P[j,i])*V_y[j,i]
                     +(T[j-1,i]+T[j,i])*(P[j,i]-P[j-1,i])/dy)/2
        T_y[j,i] = (T[j,i]-T[j-1,i])/dy
    end

    for j in 2:ny
        J_x[j,nx+1] = -( (P[j,nx-1]+P[j,2])*V_x[j,2]
                        +(T[j,nx-1]+T[j,2])*(P[j,2]-P[j,nx-1])/dx)/2
        J_x[j,1] = -( (P[j,nx-1]+P[j,2])*V_x[j,2]
                     +(T[j,nx-1]+T[j,2])*(P[j,2]-P[j,nx-1])/dx)/2
        T_x[j,1] = (T[j,1]-T[j,nx-1])/dx
        T_x[j,nx+1] = (T[j,2]-T[j,nx])/dx

        J_y[j,1] = -( (P[j-1,1]+P[j,1])*V_y[j,1]
                     +(T[j-1,1]+T[j,1])*(P[j,2]-P[j-1,1])/dy)/2
        J_y[j,nx+1] = -( (P[j-1,2]+P[j,2])*V_y[j,2]
                        +(T[j-1,2]+T[j,2])*(P[j,2]-P[j-1,2])/dy)/2
        T_y[j,1] = (T[j,1]-T[j-1,1])/dy
        T_y[j,nx+1] = (T[j,2]-T[j-1,2])/dy
    end

    T_x[1,2:nx] .= (T[1,2:nx]-T[1,1:nx-1])/dx
    T_x[1,nx+1] = (T[1,2]-T[1,1])/dx
    T_x[1,1] = (T[1,1]-T[1,nx-1])/dx
    T_x[ny+1,1:nx+1] = zero(eltype(J_y))

    T_y[1,1:nx] .= (T[1,1:nx]-T0)/dy
    T_y[1,nx+1] = (T[1,2]-T0)/dy
    T_y[ny+1,1:nx] .= (T0-T[ny,1:nx])/dy
    T_y[ny+1,nx+1] = (T0-T[2,2])/dy

    J_x[1,:] = zero(eltype(J_y))
    J_x[ny+1,:] = zero(eltype(J_y))
    J_y[1,:] = zero(eltype(J_y))
    J_y[ny+1,:] = zero(eltype(J_y))


    du[1:nx*ny] .= ((J_x[1:ny,1:nx]-J_x[1:ny,2:nx+1])/dy + (J_y[1:ny,1:nx]-J_y[2:ny+1,1:nx])/dx)[:]

    # du[(nx*ny+1):2nx*ny] .= ((J_x[1:ny,1:nx].*Vxshift[1:ny,1:nx]-J_x[1:ny,2:nx+1].*Vxshift[1:ny,2:nx+1]
    #                           -coupling*(T_x[1:ny,1:nx]-T_x[1:ny,2:nx+1]))/dy
    #                          +(J_y[1:ny,1:nx].*Vyshift[1:ny,1:nx]-J_y[2:ny+1,1:nx].*Vyshift[2:ny+1,1:nx]
    #                           -coupling*(T_y[1:ny,1:nx]-T_y[2:ny+1,1:nx]))/dx)[:]
    du[(nx*ny+1):2nx*ny] .= A*(((J_x[1:ny,1:nx].*Vxshift[1:ny,1:nx]-J_x[1:ny,2:nx+1].*Vxshift[1:ny,2:nx+1]
                               -coupling*(T_x[1:ny,1:nx]-T_x[1:ny,2:nx+1]))/dy
                             +(J_y[1:ny,1:nx].*Vyshift[1:ny,1:nx]-J_y[2:ny+1,1:nx].*Vyshift[2:ny+1,1:nx]
                               -coupling*(T_y[1:ny,1:nx]-T_y[2:ny+1,1:nx]))/dx))[:].-A*du[1:nx*ny].*V[:]
end

flux!(du,u,params,t) = flux!(du,u,params...)

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
#     energy_flux = A*(((J_x22*(V22+V12)/2 - J_x12*Vxshift[1:ny,2:nx+1]
#                                -coupling*(T_x[1:ny,1:nx]-T_x[1:ny,2:nx+1]))/dy
#                              +(J_y[1:ny,1:nx].*Vyshift[1:ny,1:nx]-J_y[2:ny+1,1:nx].*Vyshift[2:ny+1,1:nx]
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

    J_x = Array{elT}(ny,nx)
    J_y = Array{elT}(ny,nx)
    for i in 1:nx, j in 1:ny
        J_x[j,i] = -( (P[j,i-1]+P[j,i])*V_x[j,i]
                     +(T[j,i-1]+T[j,i])*(P[j,i]-P[j,i-1])/dx)/2
        J_y[j,i] = -( (P[j-1,i]+P[j,i])*V_y[j,i]
                     +(T[j-1,i]+T[j,i])*(P[j,i]-P[j-1,i])/dy)/2
    end

    J_x,J_y
end
