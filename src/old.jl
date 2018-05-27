density_flux_code =
"""
function density_flux!(::Type{Val{:jac}},jac,P,Pmat,Tmat,Jx,Jy,V_x,V_y,dx,dy)
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
        jac[diag_0_indices[row]] = $(dP[1])
        jac[diag_m1_indices[row]] = $(dP[2])
        jac[diag_mnx_indices[row]] = $(dP[3])
        jac[diag_p1_indices[row]] = $(dP[4])
        jac[diag_pnx_indices[row]] = $(dP[5])
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
"""
clipboard(density_flux_code)
##
dT = Array{String}(5)
for (index,var) in enumerate([T_22,T_12,T_21,T_32,T_23])
    var = diff(density_flux,var) |> simplify |> string
    for replacement in replacements
        var = replace(var,replacement...)
    end
    var = replace(var," ","")
    dT[index] = var
end
dT

temperature_flux_code =
"""
function temperature_flux!(::Type{Val{:jac}},jac,T,Pmat,Tmat,Jx,Jy,V_x,V_y,dx,dy)
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
        jac[diag_0_indices[row]] = $(dT[1])
        jac[diag_m1_indices[row]] = $(dT[2])
        jac[diag_mnx_indices[row]] = $(dT[3])
        jac[diag_p1_indices[row]] = $(dT[4])
        jac[diag_pnx_indices[row]] = $(dT[5])
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
"""
clipboard(temperature_flux_code)


function stencil_indices!(inds, i, j, nn)::Array{Bool, 1}
    inds[:] .= true
    j < 1 && (inds[1:3] .= false)
    j > nn && (inds[7:9] .= false)
    # i <= 1 && (inds[[1,4]] .= false)
    # i >= nn && (inds[[6,9]] .= false)
    inds
end

function wrap_index(index,i,j,nx,ny)
    # 1 <= j <= ny && return index
    # i < 1 && (index = index+nx+1)
    # i > nx && (index = index-nx-1)
    1 <= index <= nx*ny ? index : mod(index,nx)+1
end

function array_index(row,i,j,nx,ny)
    ((1 <= i <= nx) && (1 <= j <= ny)) && return sub2ind((nx,ny),i,j)
    i == 0 && return sub2ind((nx,ny),nx,mod(j,ny)+1)
    i == nx+1 && return sub2ind((nx,ny),1,mod(j,ny)+1)
    j == 0 && return row
    j == ny+1 && return row
    # i == -1 &&
    # i < 1 && return mod(sub2ind((nx,ny),i,j),nx*ny)+1
    # i > nx && return mod(-sub2ind((nx,ny),i,j),nx)+1
    # 1 <= row <= nx*ny ? row : mod(row,nx)+1
end

# function density_flux!(::Type{Val{:jac}},jac,P,Pmat,Tmat,Jx,Jy,V_x,V_y,dx,dy)
#     # @show "hi"
#     ny = length(indices(Tmat)[1])-2
#     nx = length(indices(Tmat)[2])-2
#     inds = Array{Bool}(9)
#     arrayinds = Array{Int64}(9)
#     jac[:,:] = 0.0
#     for row in 1:nx*ny
#         i,j = ind2sub((nx,ny),row)
#         # arrayinds[:] .= Int64[wrap_index(sub2ind((nx,ny),ii,jj),ii,jj,nx,ny) for ii in (i-1):(i+1),
#         #                                                                    jj in (j-1):(j+1)][:]
#         arrayinds[:] .= Int64[array_index(row,ii,jj,nx,ny) for ii in (i-1):(i+1),
#                                                                jj in (j-1):(j+1)][:]
#         # stencil_indices!(inds,i,j,nx)  # TODO make this work for nx != ny.
#         # inds = [1 <= jj <= ny for ii in (i-1):(i+1), jj in (j-1):(j+1)][:]
#         inds = fill(true,9)
#         jac[row,arrayinds[inds]] .= [0.0;
#                                (Tmat[i,j-1]+Tmat[i,j]-V_y[i,j]*dy)/(2*dx*dy);
#                                0.0;
#                                (Tmat[i-1,j]+Tmat[i,j]-V_x[i,j]*dx)/(2*dx*dy);
#                                -(Tmat[i-1,j]+Tmat[i,j-1]+4*Tmat[i,j]+Tmat[i,j+1]+Tmat[i+1,j]+dx*(V_x[i,j]-V_x[i+1,j])+dy*(V_y[i,j]-V_y[i,j+1]))/(2*dx*dy);
#                                (Tmat[i,j]+Tmat[i+1,j]+V_x[i+1,j]*dx)/(2*dx*dy);
#                                0.0;
#                                (Tmat[i,j]+Tmat[i,j+1]+V_y[i,j+1]*dy)/(2*dx*dy);
#                                0.0][inds]
#     end
#     # jac[:,1:ny:nx*ny] = 0
#     jac[diagind(jac,1)[nx:nx:nx*ny-nx+1]] = 0
#     # jac[end,1] = 0
#     jac
# end
#
# density_flux!(::Type{Val{:jac}},jac,P,params,t) = density_flux!(Val{:jac},jac,P,params...)

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
