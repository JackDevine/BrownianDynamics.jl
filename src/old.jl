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
