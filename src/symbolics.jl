using SymPy

@syms(P_12,P_21,P_22,P_32,P_23,T_12,T_21,T_23,T_32,T_22,
      V_x_12,V_x_21,V_x_32,V_x_23,V_x_22,
      V_y_12,V_y_21,V_y_32,V_y_23,V_y_22,real=true)
@syms Vxs_ij Vxs_ip1j Vys_ij Vys_ip1j Jx_ij Jx_ip1j Jy_ij Jy_ip1j Jy_ijp1 dx dy A C real=true

Jx_ij = -((P_12+P_22)*V_x_22+(T_12+T_22)*(P_22-P_12)/dx)/2
Jx_ip1j = -((P_22+P_32)*V_x_32+(T_22+T_32)*(P_32-P_22)/dx)/2

Jy_ij = -((P_21+P_22)*V_y_22+(T_21+T_22)*(P_22-P_21)/dy)/2
Jy_ijp1 = -((P_22+P_23)*V_y_23+(T_22+T_23)*(P_23-P_22)/dy)/2

T_x_ij = -(T_22-T_12)/dx
T_x_ip1j = -(T_23-T_22)/dx

density_flux = (Jx_ij-Jx_ip1j)/dy+(Jy_ij-Jy_ijp1)/dx

energy_flux = ((Jx_ij*Vxs_ij-Jx_ip1j*Vxs_ip1j-C*(T_x_ij-T_x_ip1j))/dy
                 +((Jy_ij*Vys_ij-Jy_ip1j*Vys_ip1j-C*(T_x_ij-T_x_ip1j)))/dx)

temperature_flux = A*(energy_flux-density_flux)

replacements = [("T_22","Tmat[i,j]"),("T_21","Tmat[i,j-1]"),("T_12","Tmat[i-1,j]"),
                ("T_23","Tmat[i,j+1]"),("T_32","Tmat[i+1,j]"),
                ("P_22","Pmat[i,j]"),("P_21","Pmat[i,j-1]"),("P_12","Pmat[i-1,j]"),
                ("P_23","Pmat[i,j+1]"),("P_32","Pmat[i+1,j]"),
                ("V_x_22","V_x[i,j]"),("V_x_21","V_x[i,j-1]"),("V_x_12","V_x[i-1,j]"),
                ("V_x_23","V_x[i,j+1]"),("V_x_32","V_x[i+1,j]"),
                ("V_y_22","V_y[i,j]"),("V_y_21","V_y[i,j-1]"),("V_y_12","V_y[i-1,j]"),
                ("V_y_23","V_y[i,j+1]"),("V_y_32","V_y[i+1,j]"),
                ("C","coupling")]
##
density_stencil = Array{String}(5)
density_coupling_stencil = Array{String}(5)
temperature_coupling_stencil = Array{String}(5)
temperature_stencil = Array{String}(5)
@syms dP_P dT_P
for (index,var) in enumerate([P_22,P_12,P_21,P_32,P_23])
    dP_P = diff(density_flux,var) |> simplify |> string
    dT_P = diff(temperature_flux,var) |> simplify |> string
    for replacement in replacements
        dP_P = replace(dP_P,replacement...)
        dT_P = replace(dT_P,replacement...)
    end
    dP_P = replace(dP_P," ","")
    dT_P = replace(dT_P," ","")
    density_stencil[index] = dP_P
    temperature_coupling_stencil[index] = dT_P
end
@syms dP_T dT_T
for (index,var) in enumerate([T_22,T_12,T_21,T_32,T_23])
    dP_T = diff(density_flux,var) |> simplify |> string
    dT_T = diff(temperature_flux,var) |> simplify |> string
    for replacement in replacements
        dP_T = replace(dP_T,replacement...)
        dT_T = replace(dT_T,replacement...)
    end
    dP_T = replace(dP_T," ","")
    dT_T = replace(dT_T," ","")
    density_coupling_stencil[index] = dP_T
    temperature_stencil[index] = dT_T
end

function_names = ["density_flux!","density_coupling!","temperature_coupling!","temperature_flux!"]
stencils = [density_stencil,density_coupling_stencil,
            temperature_coupling_stencil,temperature_stencil]
code = Array{String}(4)
for i in 1:4
    function_name = function_names[i]
    stencil = stencils[i]
    code[i] =
    """
    function $(function_name)(::Type{Val{:jac}},jac,P,Pmat,Tmat,Jx,Jy,V_x,V_y,dx,dy)
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
            jac[diag_0_indices[row]] = $(stencil[1])
            jac[diag_m1_indices[row]] = $(stencil[2])
            jac[diag_mnx_indices[row]] = $(stencil[3])
            jac[diag_p1_indices[row]] = $(stencil[4])
            jac[diag_pnx_indices[row]] = $(stencil[5])
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
end
