"""
   flux!(du,u,A,coupling,T0,V,Vxshift,Vyshift,V_x,V_y,dx,dy)

Compute the flux `du` going out of each cell in `u`, where `u` represents `[P[:];U[:]]`,
where `P[j,i]` is the integrated probability distribution of the `i`, `j`th. `U` is the
energy density. The parameters are as follows:
  * A: Dimensionless inverse heat capacity
  * coupling: $\\frac{B}{A}$
  * T0: Boundary temperature
  * V[j,i]: Potential at the point $x_i$, $y_j$
  * Vxshift: Potential at the point $x_i-\Delta x$, $y_j$
  * Vyshift: Potential at the point $x_i$, $y_j-\Delta y$
  ...
"""
function flux!(du,u,coupling,T0,Vxshift,Vyshift,V_x,V_y,dx,dy)
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


    @. du[1:nx*ny] = ((J_x[1:ny,1:nx]-J_x[1:ny,2:nx+1])/dy + (J_y[1:ny,1:nx]-J_y[2:ny+1,1:nx])/dx)[:]

    # du[(nx*ny+1):2nx*ny] .= ((J_x[1:ny,1:nx].*Vxshift[1:ny,1:nx]-J_x[1:ny,2:nx+1].*Vxshift[1:ny,2:nx+1]
    #                           -coupling*(T_x[1:ny,1:nx]-T_x[1:ny,2:nx+1]))/dy
    #                          +(J_y[1:ny,1:nx].*Vyshift[1:ny,1:nx]-J_y[2:ny+1,1:nx].*Vyshift[2:ny+1,1:nx]
    #                           -coupling*(T_y[1:ny,1:nx]-T_y[2:ny+1,1:nx]))/dx)[:]
    @. du[(nx*ny+1):2nx*ny] = A*(((J_x[1:ny,1:nx].*Vxshift[1:ny,1:nx]-J_x[1:ny,2:nx+1].*Vxshift[1:ny,2:nx+1]
                               -coupling*(T_x[1:ny,1:nx]-T_x[1:ny,2:nx+1]))/dy
                             +(J_y[1:ny,1:nx].*Vyshift[1:ny,1:nx]-J_y[2:ny+1,1:nx].*Vyshift[2:ny+1,1:nx]
                               -coupling*(T_y[1:ny,1:nx]-T_y[2:ny+1,1:nx]))/dx))[:]-A*du[1:nx*ny].*V[:]
end
