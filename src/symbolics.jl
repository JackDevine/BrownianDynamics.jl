using SymPy

@syms(P_12,P_21,P_22,P_32,P_23,T_12,T_21,T_23,T_32,T_22,
      V_x_12,V_x_21,V_x_32,V_x_23,V_x_22,
      V_y_12,V_y_21,V_y_32,V_y_23,V_y_22,real=true)
@syms Vxs_ij Vxs_ip1j Jx_ij Jx_ip1j Jy_ij Jy_ijp1 dx dy A C real=true

Jx_ij = -((P_12+P_22)*V_x_22+(T_12+T_22)*(P_22-P_12)/dx)/2
Jx_ip1j = -((P_22+P_32)*V_x_32+(T_22+T_32)*(P_32-P_22)/dx)/2

Jy_ij = -((P_21+P_22)*V_y_22+(T_21+T_22)*(P_22-P_21)/dy)/2
Jy_ijp1 = -((P_22+P_23)*V_y_23+(T_22+T_23)*(P_23-P_22)/dy)/2

T_x_ij = -(T22-T12)/dx
T_x_ip1j = -()/dx

density_flux = (Jx_ij-Jx_ip1j)/dy+(Jy_ij-Jy_ijp1)/dx

energy_flux = A*((Jx_ij*Vxs_ij-Jx_ip1j*Vxs_ip1j-C*(T_x_ij-T_x_ip1j))/dy
                 +((Jy_ij*Vys_ij-Jy_ip1j*Vys_ip1j-C*(T_x_ij-T_x_ip1j)))/dx)

replacements = [("T_22","Tmat[i,j]"),("T_21","Tmat[i,j-1]"),("T_12","Tmat[i-1,j]"),
                ("T_23","Tmat[i,j+1]"),("T_32","Tmat[i+1,j]"),
                ("V_x_22","V_x[i,j]"),("V_x_21","V_x[i,j-1]"),("V_x_12","V_x[i-1,j]"),
                ("V_x_23","V_x[i,j+1]"),("V_x_32","V_x[i+1,j]"),
                ("V_y_22","V_y[i,j]"),("V_y_21","V_y[i,j-1]"),("V_y_12","V_y[i-1,j]"),
                ("V_y_23","V_y[i,j+1]"),("V_y_32","V_y[i+1,j]")]
dP = Array{String}(5)
for (index,var) in enumerate([P_22,P_12,P_21,P_32,P_23])
    var = diff(density_flux,var) |> simplify |> string
    for replacement in replacements
        var = replace(var,replacement...)
    end
    var = replace(var," ","")
    dP[index] = var
end
dP
##
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
clipboard(code)
##

# [P_22,P_12,P_21,P_32,P_23]
code =
"""
function density_flux!(::Type{Val{:jac}},jac,P,Pmat,Tmat,Jx,Jy,V_x,V_y,dx,dy)
    ny = length(indices(Tmat)[1])-2
    nx = length(indices(Tmat)[2])-2
    inds = Array{Bool}(9)
    arrayinds = Array{Int64}(9)
    for row in 1:nx*ny
        i,j = ind2sub((nx,ny),row)
        arrayinds = Int64[sub2ind((nx,ny),ii,jj) for ii in (i-1):(i+1),
                                                     jj in (j-1):(j+1)]
        stencil_indices!(inds,i,j,nx)  # TODO make this work for nx != ny.
        jac[row,arrayinds[inds]] .= [0.0;
                                     $(dP[2]);
                                     0.0;
                                     $(dP[3]);
                                     $(dP[1]);
                                     $(dP[4]);
                                     0.0;
                                     $(dP[5]);
                                     0.0][inds]
    end
    jac
end
"""

clipboard(code)
# ny = length(indices(Tmat)[1])-2
# nx = length(indices(Tmat)[2])-2
# diag_0_indices = diagind(jac,0)  # dPij.
# diag_m1_indices = diagind(jac,-1)  # dPijm1.
# # diag_mnxm1_indices = diagind(jac,-nx)  # dPim1jm1.
# diag_mnx_indices = diagind(jac,-nx)  # dPim1j.
# # diag_mnxp1_indices = diagind(jac,-nx)  # dPim1jp1.
# diag_p1_indices = diagind(jac,1)  # dPijp1.
# diag_pnx_indices = diagind(jac,nx)  # dPip1j
# for row in 1:nx*ny
#     i,j = ind2sub((ny,nx),row)
#     jac[diag_0_indices[row]] = -(T[j,i-1]+T[j-1,i]+4*T[j,i]+T[j,i+1]+T[j+1,i]+dx*(V_x[j,i]-V_x[j+1,i])+dy*(V_x[j,i]-V_x[j,i+1]))/(2*dx*dy)
# end
# for row in 2:(nx*ny)
#     i,j = ind2sub((ny,nx),row-1)
#     jac[diag_m1_indices[row-1]] = (T[j-1,i]+T[j,i]-V_x[j,i]*dy)/(2*dx*dy)
# end
# for row in (nx+1):(nx*ny)
#     i,j = ind2sub((ny,nx),row-nx)
#     jac[diag_mnx_indices[row-nx]] = (T[j,i-1]+T[j,i]-V_x[j,i]*dx)/(2*dx*dy)
# end
# for row in 1:(nx*ny-1)
#     i,j = ind2sub((ny,nx),row)
#     jac[diag_p1_indices[row]] = (T[j,i]+T[j+1,i]+V_x[j+1,i]*dx)/(2*dx*dy)
# end
# for row in 1:(nx*ny-nx-1)
#     i,j = ind2sub((ny,nx),row)
#     jac[diag_pnx_indices[row]] = (T[j,i]+T[j,i+1]+V_x[j,i+1]*dy)/(2*dx*dy)
# end
# jac
