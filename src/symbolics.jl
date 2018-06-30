#=
This script will create code for the Jacobian of the finite volume discretization. If you
run this script, then the code which is commented out at the bottom of the script will be
placed into your clipboard. The code can then be used to calculate Jacobians. We have four
functions that get created,
density_flux!             The density part of the flux differentiated w.r.t. P_ij
density_coupling!         The density part of the flux differentiated w.r.t. T_ij
temperature_coupling!     The temperature part of the flux differentiated w.r.t. P_ij
temperature_flux!         The temperature part of the flux differentiated w.r.t. T_ij

The finite volume discretization is local, so the flux at a given point (i,j) only depends
on the variables at that point and the points surrounding it. We use a five point stencil
of the form:
            u_i,j-1
    u_i-1,j u_ij   u_i+1j
            u_ij+1
Throughout the symbolic section, we will denote points on this stencil as follows:
         u_21
    u_12 u_22 u_23
         u_32
We will have to calculate the flux of the probability distribution and the flux of the
temperature distribution. Both of these depend on the probability distribution, the
temperature and the potential. We have the following notations for discretised variables:

P_ij:    The discrete probability distribution at stencil point i,j.
T_ij:    The discrete temperature at stencil point i,j.
T_x_ij:  The x gradient of the temperature at the left edge of the i,j cell.
T_y_ij:  The y gradient of the temperature at the bottom edge of the i,j cell.
V_x_ij:  The x derivative of the potential at the left edge of the i,j cell.
V_y_ij:  The y derivative of the potential at the bottom edge of the i,j cell.
Vxs_ij:  The potential at the left edge of the i,j cell. (i.e. the x shifted potential.)
Vys_ij:  The potential at the bottom edge of the i,j cell. (i.e. the y shifted potential.)
=#

using SymPy
# Create the symbolic variables.
@syms(P_12,P_21,P_22,P_32,P_23,T_12,T_21,T_23,T_32,T_22,
      V_x_12,V_x_21,V_x_32,V_x_23,V_x_22,
      V_y_12,V_y_21,V_y_32,V_y_23,V_y_22,V_22,real=true)
@syms(Vxs_ij,Vxs_ip1j,Vys_ij,Vys_ijp1,Jx_ij,Jx_ip1j,
      Jy_ij,Jy_ip1j,Jy_ijp1,dx,dy,A,C,real=true)

Jx_ij = -((P_12+P_22)*V_x_22+(T_12+T_22)*(P_22-P_12)/dx)/2
Jx_ip1j = -((P_22+P_32)*V_x_32+(T_22+T_32)*(P_32-P_22)/dx)/2

Jy_ij = -((P_21+P_22)*V_y_22+(T_21+T_22)*(P_22-P_21)/dy)/2
Jy_ijp1 = -((P_22+P_23)*V_y_23+(T_22+T_23)*(P_23-P_22)/dy)/2

T_x_ij = (T_22-T_12)/dx
T_x_ip1j = (T_32-T_22)/dx
T_y_ij = (T_22-T_21)/dy
T_y_ijp1 = (T_23-T_22)/dy
# The definitions for the fluxes are based on the fluxes from the function
# `flux!(du,u,params...)`.
density_flux = (Jx_ij-Jx_ip1j)/dy+(Jy_ij-Jy_ijp1)/dx
energy_flux = ((Jx_ij*Vxs_ij-Jx_ip1j*Vxs_ip1j-C*(T_x_ij-T_x_ip1j))/dy
                 +((Jy_ij*Vys_ij-Jy_ijp1*Vys_ijp1-C*(T_y_ij-T_y_ijp1)))/dx)
temperature_flux = A*(energy_flux-density_flux*V_22)
# Now we want to turn the symbolic expressions into Julia code. To do this, we will
# replace the stencil variables with their corresponding array variables.
replacements = [("T_22","Tmat[i,j]"),("T_21","Tmat[i,j-1]"),("T_12","Tmat[i-1,j]"),
                ("T_23","Tmat[i,j+1]"),("T_32","Tmat[i+1,j]"),
                ("P_22","Pmat[i,j]"),("P_21","Pmat[i,j-1]"),("P_12","Pmat[i-1,j]"),
                ("P_23","Pmat[i,j+1]"),("P_32","Pmat[i+1,j]"),
                ("V_x_22","V_x[i,j]"),("V_x_21","V_x[i,j-1]"),("V_x_12","V_x[i-1,j]"),
                ("V_x_23","V_x[i,j+1]"),("V_x_32","V_x[i+1,j]"),
                ("V_y_22","V_y[i,j]"),("V_y_21","V_y[i,j-1]"),("V_y_12","V_y[i-1,j]"),
                ("V_y_23","V_y[i,j+1]"),("V_y_32","V_y[i+1,j]"),
                ("Vxs_ip1j","Vxshift[i+1,j]"),("Vxs_ij","Vxshift[i,j]"),
                ("Vys_ijp1","Vyshift[i,j+1]"),("Vys_ij","Vyshift[i,j]"),
                ("V_22","V[i,j]"),
                ("C","coupling")]
# These stencils correspond to the functions:
#    density_flux!
#    density_coupling!
#    temperature_coupling!
#    temperature_flux!
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

function_names = ["density_flux!","density_coupling!",
                  "temperature_coupling!","temperature_flux!"]
stencils = [density_stencil,density_coupling_stencil,
            temperature_coupling_stencil,temperature_stencil]
code = Array{String}(4)
for i in 1:4
    function_name = function_names[i]
    stencil = stencils[i]
    code[i] =
    """
    function $(function_name)(::Type{Val{:jac}},jac,u,Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,
                              V_x,V_y,dx,dy,A,coupling)
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

        arrayinds = Array{Int64}(5)
        inds = Array{Int64}(5)
        for row in 1:nx*ny
            i,j = ind2sub((nx,ny),row)
            inds[:] .= stencil_indices((nx,ny),row)
            jac[row,inds] .= [$(stencil[1]),$(stencil[2]),$(stencil[3]),
                              $(stencil[4]),$(stencil[5])]
        end
        jac[diagind(jac,nx*ny-1)] = 0
        jac[diagind(jac,-nx*ny+1)] = 0
        jac[diagind(jac,nx*ny-nx)] = 0
        jac[diagind(jac,-nx*ny+nx)] = 0
        jac
    end

    $(function_name)(::Type{Val{:jac}},jac,u,params,t) = $(function_name)(Val{:jac},jac,u,params...)
    """
end

join(code,"\n") |> clipboard
## Results.
#
# function density_flux!(::Type{Val{:jac}},jac,u,Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,
#                           V_x,V_y,dx,dy,A,coupling)
#     ny = length(indices(Tmat)[1])-2
#     nx = length(indices(Tmat)[2])-2
#     Pmat[1:nx,1:ny] .= reshape(u[1:nx*ny],nx,ny)
#     # Periodicity in the x direction.
#     Pmat[0,1:ny] .= Pmat[nx-1,1:ny]
#     Pmat[nx+1,1:ny] .= Pmat[2,1:ny]
#
#     Tmat[1:nx,1:ny] .= reshape(u[(nx*ny+1):end],nx,ny)
#     # Periodicity in the x direction.
#     Tmat[0,1:ny] .= Tmat[nx-1,1:ny]
#     Tmat[nx+1,1:ny] .= Tmat[2,1:ny]
#
#     arrayinds = Array{Int64}(5)
#     inds = Array{Int64}(5)
#     for row in 1:nx*ny
#         i,j = ind2sub((nx,ny),row)
#         inds[:] .= stencil_indices((nx,ny),row)
#         jac[row,inds] .= [-(Tmat[i-1,j]+Tmat[i,j-1]+4*Tmat[i,j]+Tmat[i,j+1]+Tmat[i+1,j]+dx*(V_x[i,j]-V_x[i+1,j])+dy*(V_y[i,j]-V_y[i,j+1]))/(2*dx*dy),(Tmat[i-1,j]+Tmat[i,j]-V_x[i,j]*dx)/(2*dx*dy),(Tmat[i,j-1]+Tmat[i,j]-V_y[i,j]*dy)/(2*dx*dy),
#                           (Tmat[i,j]+Tmat[i+1,j]+V_x[i+1,j]*dx)/(2*dx*dy),(Tmat[i,j]+Tmat[i,j+1]+V_y[i,j+1]*dy)/(2*dx*dy)]
#     end
#     jac[diagind(jac,nx*ny-1)] = 0
#     jac[diagind(jac,-nx*ny+1)] = 0
#     jac[diagind(jac,nx*ny-nx)] = 0
#     jac[diagind(jac,-nx*ny+nx)] = 0
#     jac
# end
#
# density_flux!(::Type{Val{:jac}},jac,u,params,t) = density_flux!(Val{:jac},jac,u,params...)
#
# function density_coupling!(::Type{Val{:jac}},jac,u,Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,
#                           V_x,V_y,dx,dy,A,coupling)
#     ny = length(indices(Tmat)[1])-2
#     nx = length(indices(Tmat)[2])-2
#     Pmat[1:nx,1:ny] .= reshape(u[1:nx*ny],nx,ny)
#     # Periodicity in the x direction.
#     Pmat[0,1:ny] .= Pmat[nx-1,1:ny]
#     Pmat[nx+1,1:ny] .= Pmat[2,1:ny]
#
#     Tmat[1:nx,1:ny] .= reshape(u[(nx*ny+1):end],nx,ny)
#     # Periodicity in the x direction.
#     Tmat[0,1:ny] .= Tmat[nx-1,1:ny]
#     Tmat[nx+1,1:ny] .= Tmat[2,1:ny]
#
#     arrayinds = Array{Int64}(5)
#     inds = Array{Int64}(5)
#     for row in 1:nx*ny
#         i,j = ind2sub((nx,ny),row)
#         inds[:] .= stencil_indices((nx,ny),row)
#         jac[row,inds] .= [(Pmat[i-1,j]+Pmat[i,j-1]-4*Pmat[i,j]+Pmat[i,j+1]+Pmat[i+1,j])/(2*dx*dy),(Pmat[i-1,j]-Pmat[i,j])/(2*dx*dy),(Pmat[i,j-1]-Pmat[i,j])/(2*dx*dy),
#                           (-Pmat[i,j]+Pmat[i+1,j])/(2*dx*dy),(-Pmat[i,j]+Pmat[i,j+1])/(2*dx*dy)]
#     end
#     jac[diagind(jac,nx*ny-1)] = 0
#     jac[diagind(jac,-nx*ny+1)] = 0
#     jac[diagind(jac,nx*ny-nx)] = 0
#     jac[diagind(jac,-nx*ny+nx)] = 0
#     jac
# end
#
# density_coupling!(::Type{Val{:jac}},jac,u,params,t) = density_coupling!(Val{:jac},jac,u,params...)
#
# function temperature_coupling!(::Type{Val{:jac}},jac,u,Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,
#                           V_x,V_y,dx,dy,A,coupling)
#     ny = length(indices(Tmat)[1])-2
#     nx = length(indices(Tmat)[2])-2
#     Pmat[1:nx,1:ny] .= reshape(u[1:nx*ny],nx,ny)
#     # Periodicity in the x direction.
#     Pmat[0,1:ny] .= Pmat[nx-1,1:ny]
#     Pmat[nx+1,1:ny] .= Pmat[2,1:ny]
#
#     Tmat[1:nx,1:ny] .= reshape(u[(nx*ny+1):end],nx,ny)
#     # Periodicity in the x direction.
#     Tmat[0,1:ny] .= Tmat[nx-1,1:ny]
#     Tmat[nx+1,1:ny] .= Tmat[2,1:ny]
#
#     arrayinds = Array{Int64}(5)
#     inds = Array{Int64}(5)
#     for row in 1:nx*ny
#         i,j = ind2sub((nx,ny),row)
#         inds[:] .= stencil_indices((nx,ny),row)
#         jac[row,inds] .= [-A*(-V[i,j]*(Tmat[i-1,j]+Tmat[i,j-1]+4*Tmat[i,j]+Tmat[i,j+1]+Tmat[i+1,j]+dx*(V_x[i,j]-V_x[i+1,j])+dy*(V_y[i,j]-V_y[i,j+1]))+Vxshift[i,j]*(Tmat[i-1,j]+Tmat[i,j]+V_x[i,j]*dx)+Vxshift[i+1,j]*(Tmat[i,j]+Tmat[i+1,j]-V_x[i+1,j]*dx)+Vyshift[i,j]*(Tmat[i,j-1]+Tmat[i,j]+V_y[i,j]*dy)+Vyshift[i,j+1]*(Tmat[i,j]+Tmat[i,j+1]-V_y[i,j+1]*dy))/(2*dx*dy),-A*(V[i,j]-Vxshift[i,j])*(Tmat[i-1,j]+Tmat[i,j]-V_x[i,j]*dx)/(2*dx*dy),-A*(V[i,j]-Vyshift[i,j])*(Tmat[i,j-1]+Tmat[i,j]-V_y[i,j]*dy)/(2*dx*dy),
#                           -A*(V[i,j]-Vxshift[i+1,j])*(Tmat[i,j]+Tmat[i+1,j]+V_x[i+1,j]*dx)/(2*dx*dy),-A*(V[i,j]-Vyshift[i,j+1])*(Tmat[i,j]+Tmat[i,j+1]+V_y[i,j+1]*dy)/(2*dx*dy)]
#     end
#     jac[diagind(jac,nx*ny-1)] = 0
#     jac[diagind(jac,-nx*ny+1)] = 0
#     jac[diagind(jac,nx*ny-nx)] = 0
#     jac[diagind(jac,-nx*ny+nx)] = 0
#     jac
# end
#
# temperature_coupling!(::Type{Val{:jac}},jac,u,params,t) = temperature_coupling!(Val{:jac},jac,u,params...)
#
# function temperature_flux!(::Type{Val{:jac}},jac,u,Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,
#                           V_x,V_y,dx,dy,A,coupling)
#     ny = length(indices(Tmat)[1])-2
#     nx = length(indices(Tmat)[2])-2
#     Pmat[1:nx,1:ny] .= reshape(u[1:nx*ny],nx,ny)
#     # Periodicity in the x direction.
#     Pmat[0,1:ny] .= Pmat[nx-1,1:ny]
#     Pmat[nx+1,1:ny] .= Pmat[2,1:ny]
#
#     Tmat[1:nx,1:ny] .= reshape(u[(nx*ny+1):end],nx,ny)
#     # Periodicity in the x direction.
#     Tmat[0,1:ny] .= Tmat[nx-1,1:ny]
#     Tmat[nx+1,1:ny] .= Tmat[2,1:ny]
#
#     arrayinds = Array{Int64}(5)
#     inds = Array{Int64}(5)
#     for row in 1:nx*ny
#         i,j = ind2sub((nx,ny),row)
#         inds[:] .= stencil_indices((nx,ny),row)
#         jac[row,inds] .= [-A*(8*coupling+V[i,j]*(Pmat[i-1,j]+Pmat[i,j-1]-4*Pmat[i,j]+Pmat[i,j+1]+Pmat[i+1,j])-Vxshift[i,j]*(Pmat[i-1,j]-Pmat[i,j])+Vxshift[i+1,j]*(Pmat[i,j]-Pmat[i+1,j])-Vyshift[i,j]*(Pmat[i,j-1]-Pmat[i,j])+Vyshift[i,j+1]*(Pmat[i,j]-Pmat[i,j+1]))/(2*dx*dy),A*(2*coupling-V[i,j]*(Pmat[i-1,j]-Pmat[i,j])+Vxshift[i,j]*(Pmat[i-1,j]-Pmat[i,j]))/(2*dx*dy),A*(2*coupling-V[i,j]*(Pmat[i,j-1]-Pmat[i,j])+Vyshift[i,j]*(Pmat[i,j-1]-Pmat[i,j]))/(2*dx*dy),
#                           A*(2*coupling+V[i,j]*(Pmat[i,j]-Pmat[i+1,j])-Vxshift[i+1,j]*(Pmat[i,j]-Pmat[i+1,j]))/(2*dx*dy),A*(2*coupling+V[i,j]*(Pmat[i,j]-Pmat[i,j+1])-Vyshift[i,j+1]*(Pmat[i,j]-Pmat[i,j+1]))/(2*dx*dy)]
#     end
#     jac[diagind(jac,nx*ny-1)] = 0
#     jac[diagind(jac,-nx*ny+1)] = 0
#     jac[diagind(jac,nx*ny-nx)] = 0
#     jac[diagind(jac,-nx*ny+nx)] = 0
#     jac
# end
#
# temperature_flux!(::Type{Val{:jac}},jac,u,params,t) = temperature_flux!(Val{:jac},jac,u,params...)
