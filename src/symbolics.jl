

using SymPy


@syms(P_12,P_21,P_22,P_32,P_23,T_12,T_21,T_23,T_32,T_22,
      V_x_12,V_x_21,V_x_32,V_x_23,V_x_22,
      V_y_12,V_y_21,V_y_32,V_y_23,V_y_22,real=true)
@syms Jx_ij Jx_ip1j Jy_ij Jy_ijp1 dx dy real=true



Jx_ij = -((P_12+P_22)*V_x_22+(T_12+T_22)*(P_22-P_12)/dx)/2
Jx_ip1j = -((P_22+P_32)*V_x_32+(T_22+T_32)*(P_32-P_22)/dx)/2

Jy_ij = -((P_21+P_22)*V_y_22+(T_21+T_22)*(P_22-P_21)/dy)/2
Jy_ijp1 = -((P_22+P_23)*V_y_23+(T_22+T_23)*(P_23-P_22)/dy)/2

density_flux = (Jx_ij-Jx_ip1j)/dy+(Jy_ij-Jy_ijp1)/dx

replacements = [("T_22","Tmat[i,j]"),("T_21","Tmat[i,j-1]"),("T_12","Tmat[i-1,j]"),
                ("T_23","Tmat[i,j+1]"),("T_32","Tmat[i+1,j]"),
                ("V_x_22","V_x[i,j]"),("V_x_21","V_x[i,j-1]"),("V_x_12","V_x[i-1,j]"),
                ("V_x_23","V_x[i,j+1]"),("V_x_32","V_x[i+1,j]"),
                ("V_y_22","V_y[i,j]"),("V_y_21","V_y[i,j-1]"),("V_y_12","V_y[i-1,j]"),
                ("V_y_23","V_y[i,j+1]"),("V_y_32","V_y[i+1,j]")]
dP = Array{String}(5)
for (index,var) in enumerate([P_12,P_21,P_22,P_32,P_23])
    var = diff(density_flux,var) |> simplify |> string
    for replacement in replacements
        var = replace(var,replacement...)
    end
    var = replace(var," ","")
    dP[index] = var
end
dP

code =
"""
ny = length(indices(Tmat)[1])-2
nx = length(indices(Tmat)[2])-2
diag_0_indices = diagind(jac,0)  # dPij.
diag_m1_indices = diagind(jac,-1)  # dPijm1.
# diag_mnxm1_indices = diagind(jac,-nx)  # dPim1jm1.
diag_mnx_indices = diagind(jac,-nx)  # dPim1j.
# diag_mnxp1_indices = diagind(jac,-nx)  # dPim1jp1.
diag_p1_indices = diagind(jac,1)  # dPijp1.
diag_pnx_indices = diagind(jac,nx)  # dPip1j
for row in 1:nx*ny
    i,j = ind2sub((nx,ny),row)
    jac[diag_0_indices[row]] = $(dP[3])
end
for row in 2:(nx*ny)
    i,j = ind2sub((nx,ny),row-1)
    jac[diag_m1_indices[row-1]] = $(dP[2])
end
for row in (nx+1):(nx,ny)
    i,j = ind2sub((ny,nx),row-nx)
    jac[diag_mnx_indices[row-nx]] = $(dP[1])
end
for row in 1:(nx*ny-1)
    i,j = ind2sub((nx,ny),row)
    jac[diag_p1_indices[row]] = $(dP[4])
end
for row in 1:(nx*ny-nx-1)
    i,j = ind2sub((nx,ny),row)
    jac[diag_pnx_indices[row]] = $(dP[5])
end
jac
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
