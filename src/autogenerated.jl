function density_flux!(::Type{Val{:jac}},jac,u,Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling)
    ny = length(indices(Tmat)[1])-2
    nx = length(indices(Tmat)[2])-2
    Pmat[1:nx,1:ny] .= reshape(u[1:nx*ny],nx,ny)
    # Periodicity in the x direction.
    Pmat[0,1:ny] .= Pmat[nx,1:ny]
    Pmat[nx+1,1:ny] .= Pmat[1,1:ny]

    Tmat[1:nx,1:ny] .= reshape(u[(nx*ny+1):end],nx,ny)
    # Periodicity in the x direction.
    Tmat[0,1:ny] .= Tmat[nx,1:ny]
    Tmat[nx+1,1:ny] .= Tmat[1,1:ny]

    diag_0_indices = diagind(jac,0)  # dPij.
    diag_m1_indices = [diagind(jac,nx*ny-1);diagind(jac,-1)]  # dPijm1.
    diag_mnx_indices = [diagind(jac,nx*ny-nx);diagind(jac,-nx)]  # dPim1j.
    diag_p1_indices = [diagind(jac,1);diagind(jac,-nx*ny+1)]  # dPijp1.
    diag_pnx_indices = [diagind(jac,nx);diagind(jac,-nx*ny+nx)]  # dPip1j
    for row in 1:nx*ny
        i,j = ind2sub((nx,ny),row)
        if 1<i<nx && 1<j<ny
            jac[diag_0_indices[row]] = -(Tmat[i-1,j]+Tmat[i,j-1]+4*Tmat[i,j]+Tmat[i,j+1]+Tmat[i+1,j]+dx*(V_x[i,j]-V_x[i+1,j])+dy*(V_y[i,j]-V_y[i,j+1]))/(2*dx*dy)
            jac[diag_m1_indices[row]] = (Tmat[i-1,j]+Tmat[i,j]-V_x[i,j]*dx)/(2*dx*dy)
            jac[diag_mnx_indices[row]] = (Tmat[i,j-1]+Tmat[i,j]-V_y[i,j]*dy)/(2*dx*dy)
            jac[diag_p1_indices[row]] = (Tmat[i,j]+Tmat[i+1,j]+V_x[i+1,j]*dx)/(2*dx*dy)
            jac[diag_pnx_indices[row]] = (Tmat[i,j]+Tmat[i,j+1]+V_y[i,j+1]*dy)/(2*dx*dy)
        elseif i==1
            jac[diag_0_indices[row]] = -(Tmat[i-1,j]+Tmat[i,j-1]+4*Tmat[i,j]+Tmat[i,j+1]+Tmat[i+1,j]+dx*(V_x[i,j]-V_x[i+1,j])+dy*(V_y[i,j]-V_y[i,j+1]))/(2*dx*dy)
            jac[diag_m1_indices[row]] = (Tmat[i-1,j]+Tmat[i,j]-V_x[i,j]*dx)/(2*dx*dy)
            jac[diag_mnx_indices[row+nx-1]] = (Tmat[i,j-1]+Tmat[i,j]-V_y[i,j]*dy)/(2*dx*dy)
            jac[diag_p1_indices[row]] = (Tmat[i,j]+Tmat[i+1,j]+V_x[i+1,j]*dx)/(2*dx*dy)
            jac[diag_pnx_indices[row]] = (Tmat[i,j]+Tmat[i,j+1]+V_y[i,j+1]*dy)/(2*dx*dy)
        elseif i==nx
            jac[diag_0_indices[row]] = -(Tmat[i-1,j]+Tmat[i,j-1]+4*Tmat[i,j]+Tmat[i,j+1]+Tmat[i+1,j]+dx*(V_x[i,j]-V_x[i+1,j])+dy*(V_y[i,j]-V_y[i,j+1]))/(2*dx*dy)
            jac[diag_m1_indices[row]] = (Tmat[i-1,j]+Tmat[i,j]-V_x[i,j]*dx)/(2*dx*dy)
            jac[diag_mnx_indices[row]] = (Tmat[i,j-1]+Tmat[i,j]-V_y[i,j]*dy)/(2*dx*dy)
            jac[diag_p1_indices[row]] = (Tmat[i,j]+Tmat[i+1,j]+V_x[i+1,j]*dx)/(2*dx*dy)
            jac[diag_pnx_indices[row]-nx] = (Tmat[i,j]+Tmat[i,j+1]+V_y[i,j+1]*dy)/(2*dx*dy)
        elseif j==1
            jac[diag_0_indices[row]] = -(Tmat[i-1,j]+Tmat[i,j-1]+4*Tmat[i,j]+Tmat[i,j+1]+Tmat[i+1,j]+dx*(V_x[i,j]-V_x[i+1,j])+dy*(V_y[i,j]-V_y[i,j+1]))/(2*dx*dy)
            jac[diag_m1_indices[row]] = 0
            jac[diag_mnx_indices[row]] = (Tmat[i,j-1]+Tmat[i,j]-V_y[i,j]*dy)/(2*dx*dy)
            jac[diag_p1_indices[row]] = (Tmat[i,j]+Tmat[i+1,j]+V_x[i+1,j]*dx)/(2*dx*dy)
            jac[diag_pnx_indices[row]] = (Tmat[i,j]+Tmat[i,j+1]+V_y[i,j+1]*dy)/(2*dx*dy)
        elseif j==ny
            jac[diag_0_indices[row]] = -(Tmat[i-1,j]+Tmat[i,j-1]+4*Tmat[i,j]+Tmat[i,j+1]+Tmat[i+1,j]+dx*(V_x[i,j]-V_x[i+1,j])+dy*(V_y[i,j]-V_y[i,j+1]))/(2*dx*dy)
            jac[diag_m1_indices[row]] = (Tmat[i-1,j]+Tmat[i,j]-V_x[i,j]*dx)/(2*dx*dy)
            jac[diag_mnx_indices[row]] = (Tmat[i,j-1]+Tmat[i,j]-V_y[i,j]*dy)/(2*dx*dy)
            jac[diag_p1_indices[row]] = 0
            jac[diag_pnx_indices[row]] = (Tmat[i,j]+Tmat[i,j+1]+V_y[i,j+1]*dy)/(2*dx*dy)
        end
    end
    # arrayinds = Array{Int64}(5)
    # for row in 1:nx*ny
    #     i,j = ind2sub((nx,ny),row)
    #     arrayinds[:] .= [diag_0_indices[row],diag_m1_indices[row],
    #                     diag_mnx_indices[row],diag_p1_indices[row],
    #                     diag_pnx_indices[row]]
    #     jac[arrayinds] .= [-(Tmat[i-1,j]+Tmat[i,j-1]+4*Tmat[i,j]+Tmat[i,j+1]+Tmat[i+1,j]+dx*(V_x[i,j]-V_x[i+1,j])+dy*(V_y[i,j]-V_y[i,j+1]))/(2*dx*dy),(Tmat[i-1,j]+Tmat[i,j]-V_x[i,j]*dx)/(2*dx*dy),(Tmat[i,j-1]+Tmat[i,j]-V_y[i,j]*dy)/(2*dx*dy),
    #                               (Tmat[i,j]+Tmat[i+1,j]+V_x[i+1,j]*dx)/(2*dx*dy),(Tmat[i,j]+Tmat[i,j+1]+V_y[i,j+1]*dy)/(2*dx*dy)]
    # end
    # jac[diagind(jac,nx-1)[1:nx:end-nx]] .= jac[diagind(jac,-1)[nx:nx:end]]
    # jac[diagind(jac,-nx+1)[1:nx:end-nx]] .= jac[diagind(jac,1)[nx:nx:end]]
    # jac[diagind(jac,-1)[nx:nx:end]] = 0
    # jac[diagind(jac,1)[nx:nx:end]] = 0
    # jac[diagind(jac,nx*ny-1)] = 0
    # jac[diagind(jac,-nx*ny+1)] = 0
    # jac[diagind(jac,nx*ny-nx)] = 0
    # jac[diagind(jac,-nx*ny+nx)] = 0
    jac
end

density_flux!(::Type{Val{:jac}},jac,P,params,t) = density_flux!(Val{:jac},jac,P,params...)

function density_coupling!(::Type{Val{:jac}},jac,u,Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling)
    ny = length(indices(Tmat)[1])-2
    nx = length(indices(Tmat)[2])-2
    Pmat[1:nx,1:ny] .= reshape(u[1:nx*ny],nx,ny)
    # Periodicity in the x direction.
    Pmat[0,1:ny] .= Pmat[nx,1:ny]
    Pmat[nx+1,1:ny] .= Pmat[1,1:ny]

    Tmat[1:nx,1:ny] .= reshape(u[(nx*ny+1):end],nx,ny)
    # Periodicity in the x direction.
    Tmat[0,1:ny] .= Tmat[nx,1:ny]
    Tmat[nx+1,1:ny] .= Tmat[1,1:ny]

    diag_0_indices = diagind(jac,0)  # dPij.
    diag_m1_indices = [diagind(jac,nx*ny-1);diagind(jac,-1)]  # dPijm1.
    diag_mnx_indices = [diagind(jac,nx*ny-nx);diagind(jac,-nx)]  # dPim1j.
    diag_p1_indices = [diagind(jac,1);diagind(jac,-nx*ny+1)]  # dPijp1.
    diag_pnx_indices = [diagind(jac,nx);diagind(jac,-nx*ny+nx)]  # dPip1j
    for row in 1:nx*ny
        i,j = ind2sub((nx,ny),row)
        if 1<i<nx && 1<j<ny
            jac[diag_0_indices[row]] = (Pmat[i-1,j]+Pmat[i,j-1]-4*Pmat[i,j]+Pmat[i,j+1]+Pmat[i+1,j])/(2*dx*dy)
            jac[diag_m1_indices[row]] = (Pmat[i-1,j]-Pmat[i,j])/(2*dx*dy)
            jac[diag_mnx_indices[row]] = (Pmat[i,j-1]-Pmat[i,j])/(2*dx*dy)
            jac[diag_p1_indices[row]] = (-Pmat[i,j]+Pmat[i+1,j])/(2*dx*dy)
            jac[diag_pnx_indices[row]] = (-Pmat[i,j]+Pmat[i,j+1])/(2*dx*dy)
        elseif i==1
            jac[diag_0_indices[row]] = (Pmat[i-1,j]+Pmat[i,j-1]-4*Pmat[i,j]+Pmat[i,j+1]+Pmat[i+1,j])/(2*dx*dy)
            jac[diag_m1_indices[row]] = (Pmat[i-1,j]-Pmat[i,j])/(2*dx*dy)
            jac[diag_mnx_indices[row+nx-1]] = (Pmat[i,j-1]-Pmat[i,j])/(2*dx*dy)
            jac[diag_p1_indices[row]] = (-Pmat[i,j]+Pmat[i+1,j])/(2*dx*dy)
            jac[diag_pnx_indices[row]] = (-Pmat[i,j]+Pmat[i,j+1])/(2*dx*dy)
        elseif i==nx
            jac[diag_0_indices[row]] = (Pmat[i-1,j]+Pmat[i,j-1]-4*Pmat[i,j]+Pmat[i,j+1]+Pmat[i+1,j])/(2*dx*dy)
            jac[diag_m1_indices[row]] = (Pmat[i-1,j]-Pmat[i,j])/(2*dx*dy)
            jac[diag_mnx_indices[row]] = (Pmat[i,j-1]-Pmat[i,j])/(2*dx*dy)
            jac[diag_p1_indices[row]] = (-Pmat[i,j]+Pmat[i+1,j])/(2*dx*dy)
            jac[diag_pnx_indices[row]-nx] = (-Pmat[i,j]+Pmat[i,j+1])/(2*dx*dy)
        elseif j==1
            jac[diag_0_indices[row]] = (Pmat[i-1,j]+Pmat[i,j-1]-4*Pmat[i,j]+Pmat[i,j+1]+Pmat[i+1,j])/(2*dx*dy)
            jac[diag_m1_indices[row]] = 0
            jac[diag_mnx_indices[row]] = (Pmat[i,j-1]-Pmat[i,j])/(2*dx*dy)
            jac[diag_p1_indices[row]] = (-Pmat[i,j]+Pmat[i+1,j])/(2*dx*dy)
            jac[diag_pnx_indices[row]] = (-Pmat[i,j]+Pmat[i,j+1])/(2*dx*dy)
        elseif j==ny
            jac[diag_0_indices[row]] = (Pmat[i-1,j]+Pmat[i,j-1]-4*Pmat[i,j]+Pmat[i,j+1]+Pmat[i+1,j])/(2*dx*dy)
            jac[diag_m1_indices[row]] = (Pmat[i-1,j]-Pmat[i,j])/(2*dx*dy)
            jac[diag_mnx_indices[row]] = (Pmat[i,j-1]-Pmat[i,j])/(2*dx*dy)
            jac[diag_p1_indices[row]] = 0
            jac[diag_pnx_indices[row]] = (-Pmat[i,j]+Pmat[i,j+1])/(2*dx*dy)
        end
    end
    # arrayinds = Array{Int64}(5)
    # for row in 1:nx*ny
    #     i,j = ind2sub((nx,ny),row)
    #     arrayinds[:] .= [diag_0_indices[row],diag_m1_indices[row],
    #                     diag_mnx_indices[row],diag_p1_indices[row],
    #                     diag_pnx_indices[row]]
    #     jac[arrayinds] .= [(Pmat[i-1,j]+Pmat[i,j-1]-4*Pmat[i,j]+Pmat[i,j+1]+Pmat[i+1,j])/(2*dx*dy),(Pmat[i-1,j]-Pmat[i,j])/(2*dx*dy),(Pmat[i,j-1]-Pmat[i,j])/(2*dx*dy),
    #                               (-Pmat[i,j]+Pmat[i+1,j])/(2*dx*dy),(-Pmat[i,j]+Pmat[i,j+1])/(2*dx*dy)]
    # end
    # jac[diagind(jac,nx-1)[1:nx:end-nx]] .= jac[diagind(jac,-1)[nx:nx:end]]
    # jac[diagind(jac,-nx+1)[1:nx:end-nx]] .= jac[diagind(jac,1)[nx:nx:end]]
    # jac[diagind(jac,-1)[nx:nx:end]] = 0
    # jac[diagind(jac,1)[nx:nx:end]] = 0
    # jac[diagind(jac,nx*ny-1)] = 0
    # jac[diagind(jac,-nx*ny+1)] = 0
    # jac[diagind(jac,nx*ny-nx)] = 0
    # jac[diagind(jac,-nx*ny+nx)] = 0
    jac
end

density_coupling!(::Type{Val{:jac}},jac,P,params,t) = density_coupling!(Val{:jac},jac,P,params...)

function temperature_coupling!(::Type{Val{:jac}},jac,u,Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling)
    ny = length(indices(Tmat)[1])-2
    nx = length(indices(Tmat)[2])-2
    Pmat[1:nx,1:ny] .= reshape(u[1:nx*ny],nx,ny)
    # Periodicity in the x direction.
    Pmat[0,1:ny] .= Pmat[nx,1:ny]
    Pmat[nx+1,1:ny] .= Pmat[1,1:ny]

    Tmat[1:nx,1:ny] .= reshape(u[(nx*ny+1):end],nx,ny)
    # Periodicity in the x direction.
    Tmat[0,1:ny] .= Tmat[nx,1:ny]
    Tmat[nx+1,1:ny] .= Tmat[1,1:ny]

    diag_0_indices = diagind(jac,0)  # dPij.
    diag_m1_indices = [diagind(jac,nx*ny-1);diagind(jac,-1)]  # dPijm1.
    diag_mnx_indices = [diagind(jac,nx*ny-nx);diagind(jac,-nx)]  # dPim1j.
    diag_p1_indices = [diagind(jac,1);diagind(jac,-nx*ny+1)]  # dPijp1.
    diag_pnx_indices = [diagind(jac,nx);diagind(jac,-nx*ny+nx)]  # dPip1j
    for row in 1:nx*ny
        i,j = ind2sub((nx,ny),row)
        if 1<i<nx && 1<j<ny
            jac[diag_0_indices[row]] = -A*(-V[i,j]*(Tmat[i-1,j]+Tmat[i,j-1]+4*Tmat[i,j]+Tmat[i,j+1]+Tmat[i+1,j]+dx*(V_x[i,j]-V_x[i+1,j])+dy*(V_y[i,j]-V_y[i,j+1]))+Vxshift[i,j]*(Tmat[i-1,j]+Tmat[i,j]+V_x[i,j]*dx)+Vxshift[i+1,j]*(Tmat[i,j]+Tmat[i+1,j]-V_x[i+1,j]*dx)+Vyshift[i,j]*(Tmat[i,j-1]+Tmat[i,j]+V_y[i,j]*dy)+Vyshift[i,j+1]*(Tmat[i,j]+Tmat[i,j+1]-V_y[i,j+1]*dy))/(2*dx*dy)
            jac[diag_m1_indices[row]] = -A*(V[i,j]-Vxshift[i,j])*(Tmat[i-1,j]+Tmat[i,j]-V_x[i,j]*dx)/(2*dx*dy)
            jac[diag_mnx_indices[row]] = -A*(V[i,j]-Vyshift[i,j])*(Tmat[i,j-1]+Tmat[i,j]-V_y[i,j]*dy)/(2*dx*dy)
            jac[diag_p1_indices[row]] = -A*(V[i,j]-Vxshift[i+1,j])*(Tmat[i,j]+Tmat[i+1,j]+V_x[i+1,j]*dx)/(2*dx*dy)
            jac[diag_pnx_indices[row]] = -A*(V[i,j]-Vyshift[i,j+1])*(Tmat[i,j]+Tmat[i,j+1]+V_y[i,j+1]*dy)/(2*dx*dy)
        elseif i==1
            jac[diag_0_indices[row]] = -A*(-V[i,j]*(Tmat[i-1,j]+Tmat[i,j-1]+4*Tmat[i,j]+Tmat[i,j+1]+Tmat[i+1,j]+dx*(V_x[i,j]-V_x[i+1,j])+dy*(V_y[i,j]-V_y[i,j+1]))+Vxshift[i,j]*(Tmat[i-1,j]+Tmat[i,j]+V_x[i,j]*dx)+Vxshift[i+1,j]*(Tmat[i,j]+Tmat[i+1,j]-V_x[i+1,j]*dx)+Vyshift[i,j]*(Tmat[i,j-1]+Tmat[i,j]+V_y[i,j]*dy)+Vyshift[i,j+1]*(Tmat[i,j]+Tmat[i,j+1]-V_y[i,j+1]*dy))/(2*dx*dy)
            jac[diag_m1_indices[row]] = -A*(V[i,j]-Vxshift[i,j])*(Tmat[i-1,j]+Tmat[i,j]-V_x[i,j]*dx)/(2*dx*dy)
            jac[diag_mnx_indices[row+nx-1]] = -A*(V[i,j]-Vyshift[i,j])*(Tmat[i,j-1]+Tmat[i,j]-V_y[i,j]*dy)/(2*dx*dy)
            jac[diag_p1_indices[row]] = -A*(V[i,j]-Vxshift[i+1,j])*(Tmat[i,j]+Tmat[i+1,j]+V_x[i+1,j]*dx)/(2*dx*dy)
            jac[diag_pnx_indices[row]] = -A*(V[i,j]-Vyshift[i,j+1])*(Tmat[i,j]+Tmat[i,j+1]+V_y[i,j+1]*dy)/(2*dx*dy)
        elseif i==nx
            jac[diag_0_indices[row]] = -A*(-V[i,j]*(Tmat[i-1,j]+Tmat[i,j-1]+4*Tmat[i,j]+Tmat[i,j+1]+Tmat[i+1,j]+dx*(V_x[i,j]-V_x[i+1,j])+dy*(V_y[i,j]-V_y[i,j+1]))+Vxshift[i,j]*(Tmat[i-1,j]+Tmat[i,j]+V_x[i,j]*dx)+Vxshift[i+1,j]*(Tmat[i,j]+Tmat[i+1,j]-V_x[i+1,j]*dx)+Vyshift[i,j]*(Tmat[i,j-1]+Tmat[i,j]+V_y[i,j]*dy)+Vyshift[i,j+1]*(Tmat[i,j]+Tmat[i,j+1]-V_y[i,j+1]*dy))/(2*dx*dy)
            jac[diag_m1_indices[row]] = -A*(V[i,j]-Vxshift[i,j])*(Tmat[i-1,j]+Tmat[i,j]-V_x[i,j]*dx)/(2*dx*dy)
            jac[diag_mnx_indices[row]] = -A*(V[i,j]-Vyshift[i,j])*(Tmat[i,j-1]+Tmat[i,j]-V_y[i,j]*dy)/(2*dx*dy)
            jac[diag_p1_indices[row]] = -A*(V[i,j]-Vxshift[i+1,j])*(Tmat[i,j]+Tmat[i+1,j]+V_x[i+1,j]*dx)/(2*dx*dy)
            jac[diag_pnx_indices[row]-nx] = -A*(V[i,j]-Vyshift[i,j+1])*(Tmat[i,j]+Tmat[i,j+1]+V_y[i,j+1]*dy)/(2*dx*dy)
        elseif j==1
            jac[diag_0_indices[row]] = -A*(-V[i,j]*(Tmat[i-1,j]+Tmat[i,j-1]+4*Tmat[i,j]+Tmat[i,j+1]+Tmat[i+1,j]+dx*(V_x[i,j]-V_x[i+1,j])+dy*(V_y[i,j]-V_y[i,j+1]))+Vxshift[i,j]*(Tmat[i-1,j]+Tmat[i,j]+V_x[i,j]*dx)+Vxshift[i+1,j]*(Tmat[i,j]+Tmat[i+1,j]-V_x[i+1,j]*dx)+Vyshift[i,j]*(Tmat[i,j-1]+Tmat[i,j]+V_y[i,j]*dy)+Vyshift[i,j+1]*(Tmat[i,j]+Tmat[i,j+1]-V_y[i,j+1]*dy))/(2*dx*dy)
            jac[diag_m1_indices[row]] = 0
            jac[diag_mnx_indices[row]] = -A*(V[i,j]-Vyshift[i,j])*(Tmat[i,j-1]+Tmat[i,j]-V_y[i,j]*dy)/(2*dx*dy)
            jac[diag_p1_indices[row]] = -A*(V[i,j]-Vxshift[i+1,j])*(Tmat[i,j]+Tmat[i+1,j]+V_x[i+1,j]*dx)/(2*dx*dy)
            jac[diag_pnx_indices[row]] = -A*(V[i,j]-Vyshift[i,j+1])*(Tmat[i,j]+Tmat[i,j+1]+V_y[i,j+1]*dy)/(2*dx*dy)
        elseif j==ny
            jac[diag_0_indices[row]] = -A*(-V[i,j]*(Tmat[i-1,j]+Tmat[i,j-1]+4*Tmat[i,j]+Tmat[i,j+1]+Tmat[i+1,j]+dx*(V_x[i,j]-V_x[i+1,j])+dy*(V_y[i,j]-V_y[i,j+1]))+Vxshift[i,j]*(Tmat[i-1,j]+Tmat[i,j]+V_x[i,j]*dx)+Vxshift[i+1,j]*(Tmat[i,j]+Tmat[i+1,j]-V_x[i+1,j]*dx)+Vyshift[i,j]*(Tmat[i,j-1]+Tmat[i,j]+V_y[i,j]*dy)+Vyshift[i,j+1]*(Tmat[i,j]+Tmat[i,j+1]-V_y[i,j+1]*dy))/(2*dx*dy)
            jac[diag_m1_indices[row]] = -A*(V[i,j]-Vxshift[i,j])*(Tmat[i-1,j]+Tmat[i,j]-V_x[i,j]*dx)/(2*dx*dy)
            jac[diag_mnx_indices[row]] = -A*(V[i,j]-Vyshift[i,j])*(Tmat[i,j-1]+Tmat[i,j]-V_y[i,j]*dy)/(2*dx*dy)
            jac[diag_p1_indices[row]] = 0
            jac[diag_pnx_indices[row]] = -A*(V[i,j]-Vyshift[i,j+1])*(Tmat[i,j]+Tmat[i,j+1]+V_y[i,j+1]*dy)/(2*dx*dy)
        end
    end
    # arrayinds = Array{Int64}(5)
    # for row in 1:nx*ny
    #     i,j = ind2sub((nx,ny),row)
    #     arrayinds[:] .= [diag_0_indices[row],diag_m1_indices[row],
    #                     diag_mnx_indices[row],diag_p1_indices[row],
    #                     diag_pnx_indices[row]]
    #     jac[arrayinds] .= [-A*(-V[i,j]*(Tmat[i-1,j]+Tmat[i,j-1]+4*Tmat[i,j]+Tmat[i,j+1]+Tmat[i+1,j]+dx*(V_x[i,j]-V_x[i+1,j])+dy*(V_y[i,j]-V_y[i,j+1]))+Vxshift[i,j]*(Tmat[i-1,j]+Tmat[i,j]+V_x[i,j]*dx)+Vxshift[i+1,j]*(Tmat[i,j]+Tmat[i+1,j]-V_x[i+1,j]*dx)+Vyshift[i,j]*(Tmat[i,j-1]+Tmat[i,j]+V_y[i,j]*dy)+Vyshift[i,j+1]*(Tmat[i,j]+Tmat[i,j+1]-V_y[i,j+1]*dy))/(2*dx*dy),-A*(V[i,j]-Vxshift[i,j])*(Tmat[i-1,j]+Tmat[i,j]-V_x[i,j]*dx)/(2*dx*dy),-A*(V[i,j]-Vyshift[i,j])*(Tmat[i,j-1]+Tmat[i,j]-V_y[i,j]*dy)/(2*dx*dy),
    #                               -A*(V[i,j]-Vxshift[i+1,j])*(Tmat[i,j]+Tmat[i+1,j]+V_x[i+1,j]*dx)/(2*dx*dy),-A*(V[i,j]-Vyshift[i,j+1])*(Tmat[i,j]+Tmat[i,j+1]+V_y[i,j+1]*dy)/(2*dx*dy)]
    # end
    # jac[diagind(jac,nx-1)[1:nx:end-nx]] .= jac[diagind(jac,-1)[nx:nx:end]]
    # jac[diagind(jac,-nx+1)[1:nx:end-nx]] .= jac[diagind(jac,1)[nx:nx:end]]
    # jac[diagind(jac,-1)[nx:nx:end]] = 0
    # jac[diagind(jac,1)[nx:nx:end]] = 0
    # jac[diagind(jac,nx*ny-1)] = 0
    # jac[diagind(jac,-nx*ny+1)] = 0
    # jac[diagind(jac,nx*ny-nx)] = 0
    # jac[diagind(jac,-nx*ny+nx)] = 0
    jac
end

temperature_coupling!(::Type{Val{:jac}},jac,P,params,t) = temperature_coupling!(Val{:jac},jac,P,params...)

function temperature_flux!(::Type{Val{:jac}},jac,u,Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling)
    ny = length(indices(Tmat)[1])-2
    nx = length(indices(Tmat)[2])-2
    Pmat[1:nx,1:ny] .= reshape(u[1:nx*ny],nx,ny)
    # Periodicity in the x direction.
    Pmat[0,1:ny] .= Pmat[nx,1:ny]
    Pmat[nx+1,1:ny] .= Pmat[1,1:ny]

    Tmat[1:nx,1:ny] .= reshape(u[(nx*ny+1):end],nx,ny)
    # Periodicity in the x direction.
    Tmat[0,1:ny] .= Tmat[nx,1:ny]
    Tmat[nx+1,1:ny] .= Tmat[1,1:ny]

    diag_0_indices = diagind(jac,0)  # dPij.
    diag_m1_indices = [diagind(jac,nx*ny-1);diagind(jac,-1)]  # dPijm1.
    diag_mnx_indices = [diagind(jac,nx*ny-nx);diagind(jac,-nx)]  # dPim1j.
    diag_p1_indices = [diagind(jac,1);diagind(jac,-nx*ny+1)]  # dPijp1.
    diag_pnx_indices = [diagind(jac,nx);diagind(jac,-nx*ny+nx)]  # dPip1j
    for row in 1:nx*ny
        i,j = ind2sub((nx,ny),row)
        if 1<i<nx && 1<j<ny
            jac[diag_0_indices[row]] = -A*(8*coupling+V[i,j]*(Pmat[i-1,j]+Pmat[i,j-1]-4*Pmat[i,j]+Pmat[i,j+1]+Pmat[i+1,j])-Vxshift[i,j]*(Pmat[i-1,j]-Pmat[i,j])+Vxshift[i+1,j]*(Pmat[i,j]-Pmat[i+1,j])-Vyshift[i,j]*(Pmat[i,j-1]-Pmat[i,j])+Vyshift[i,j+1]*(Pmat[i,j]-Pmat[i,j+1]))/(2*dx*dy)
            jac[diag_m1_indices[row]] = A*(2*coupling-V[i,j]*(Pmat[i-1,j]-Pmat[i,j])+Vxshift[i,j]*(Pmat[i-1,j]-Pmat[i,j]))/(2*dx*dy)
            jac[diag_mnx_indices[row]] = A*(2*coupling-V[i,j]*(Pmat[i,j-1]-Pmat[i,j])+Vyshift[i,j]*(Pmat[i,j-1]-Pmat[i,j]))/(2*dx*dy)
            jac[diag_p1_indices[row]] = A*(2*coupling+V[i,j]*(Pmat[i,j]-Pmat[i+1,j])-Vxshift[i+1,j]*(Pmat[i,j]-Pmat[i+1,j]))/(2*dx*dy)
            jac[diag_pnx_indices[row]] = A*(2*coupling+V[i,j]*(Pmat[i,j]-Pmat[i,j+1])-Vyshift[i,j+1]*(Pmat[i,j]-Pmat[i,j+1]))/(2*dx*dy)
        elseif i==1
            jac[diag_0_indices[row]] = -A*(8*coupling+V[i,j]*(Pmat[i-1,j]+Pmat[i,j-1]-4*Pmat[i,j]+Pmat[i,j+1]+Pmat[i+1,j])-Vxshift[i,j]*(Pmat[i-1,j]-Pmat[i,j])+Vxshift[i+1,j]*(Pmat[i,j]-Pmat[i+1,j])-Vyshift[i,j]*(Pmat[i,j-1]-Pmat[i,j])+Vyshift[i,j+1]*(Pmat[i,j]-Pmat[i,j+1]))/(2*dx*dy)
            jac[diag_m1_indices[row]] = A*(2*coupling-V[i,j]*(Pmat[i-1,j]-Pmat[i,j])+Vxshift[i,j]*(Pmat[i-1,j]-Pmat[i,j]))/(2*dx*dy)
            jac[diag_mnx_indices[row+nx-1]] = A*(2*coupling-V[i,j]*(Pmat[i,j-1]-Pmat[i,j])+Vyshift[i,j]*(Pmat[i,j-1]-Pmat[i,j]))/(2*dx*dy)
            jac[diag_p1_indices[row]] = A*(2*coupling+V[i,j]*(Pmat[i,j]-Pmat[i+1,j])-Vxshift[i+1,j]*(Pmat[i,j]-Pmat[i+1,j]))/(2*dx*dy)
            jac[diag_pnx_indices[row]] = A*(2*coupling+V[i,j]*(Pmat[i,j]-Pmat[i,j+1])-Vyshift[i,j+1]*(Pmat[i,j]-Pmat[i,j+1]))/(2*dx*dy)
        elseif i==nx
            jac[diag_0_indices[row]] = -A*(8*coupling+V[i,j]*(Pmat[i-1,j]+Pmat[i,j-1]-4*Pmat[i,j]+Pmat[i,j+1]+Pmat[i+1,j])-Vxshift[i,j]*(Pmat[i-1,j]-Pmat[i,j])+Vxshift[i+1,j]*(Pmat[i,j]-Pmat[i+1,j])-Vyshift[i,j]*(Pmat[i,j-1]-Pmat[i,j])+Vyshift[i,j+1]*(Pmat[i,j]-Pmat[i,j+1]))/(2*dx*dy)
            jac[diag_m1_indices[row]] = A*(2*coupling-V[i,j]*(Pmat[i-1,j]-Pmat[i,j])+Vxshift[i,j]*(Pmat[i-1,j]-Pmat[i,j]))/(2*dx*dy)
            jac[diag_mnx_indices[row]] = A*(2*coupling-V[i,j]*(Pmat[i,j-1]-Pmat[i,j])+Vyshift[i,j]*(Pmat[i,j-1]-Pmat[i,j]))/(2*dx*dy)
            jac[diag_p1_indices[row]] = A*(2*coupling+V[i,j]*(Pmat[i,j]-Pmat[i+1,j])-Vxshift[i+1,j]*(Pmat[i,j]-Pmat[i+1,j]))/(2*dx*dy)
            jac[diag_pnx_indices[row]-nx] = A*(2*coupling+V[i,j]*(Pmat[i,j]-Pmat[i,j+1])-Vyshift[i,j+1]*(Pmat[i,j]-Pmat[i,j+1]))/(2*dx*dy)
        elseif j==1
            jac[diag_0_indices[row]] = -A*(8*coupling+V[i,j]*(Pmat[i-1,j]+Pmat[i,j-1]-4*Pmat[i,j]+Pmat[i,j+1]+Pmat[i+1,j])-Vxshift[i,j]*(Pmat[i-1,j]-Pmat[i,j])+Vxshift[i+1,j]*(Pmat[i,j]-Pmat[i+1,j])-Vyshift[i,j]*(Pmat[i,j-1]-Pmat[i,j])+Vyshift[i,j+1]*(Pmat[i,j]-Pmat[i,j+1]))/(2*dx*dy)
            jac[diag_m1_indices[row]] = 0
            jac[diag_mnx_indices[row]] = A*(2*coupling-V[i,j]*(Pmat[i,j-1]-Pmat[i,j])+Vyshift[i,j]*(Pmat[i,j-1]-Pmat[i,j]))/(2*dx*dy)
            jac[diag_p1_indices[row]] = A*(2*coupling+V[i,j]*(Pmat[i,j]-Pmat[i+1,j])-Vxshift[i+1,j]*(Pmat[i,j]-Pmat[i+1,j]))/(2*dx*dy)
            jac[diag_pnx_indices[row]] = A*(2*coupling+V[i,j]*(Pmat[i,j]-Pmat[i,j+1])-Vyshift[i,j+1]*(Pmat[i,j]-Pmat[i,j+1]))/(2*dx*dy)
        elseif j==ny
            jac[diag_0_indices[row]] = -A*(8*coupling+V[i,j]*(Pmat[i-1,j]+Pmat[i,j-1]-4*Pmat[i,j]+Pmat[i,j+1]+Pmat[i+1,j])-Vxshift[i,j]*(Pmat[i-1,j]-Pmat[i,j])+Vxshift[i+1,j]*(Pmat[i,j]-Pmat[i+1,j])-Vyshift[i,j]*(Pmat[i,j-1]-Pmat[i,j])+Vyshift[i,j+1]*(Pmat[i,j]-Pmat[i,j+1]))/(2*dx*dy)
            jac[diag_m1_indices[row]] = A*(2*coupling-V[i,j]*(Pmat[i-1,j]-Pmat[i,j])+Vxshift[i,j]*(Pmat[i-1,j]-Pmat[i,j]))/(2*dx*dy)
            jac[diag_mnx_indices[row]] = A*(2*coupling-V[i,j]*(Pmat[i,j-1]-Pmat[i,j])+Vyshift[i,j]*(Pmat[i,j-1]-Pmat[i,j]))/(2*dx*dy)
            jac[diag_p1_indices[row]] = 0
            jac[diag_pnx_indices[row]] = A*(2*coupling+V[i,j]*(Pmat[i,j]-Pmat[i,j+1])-Vyshift[i,j+1]*(Pmat[i,j]-Pmat[i,j+1]))/(2*dx*dy)
        end
    end
    # arrayinds = Array{Int64}(5)
    # for row in 1:nx*ny
    #     i,j = ind2sub((nx,ny),row)
    #     arrayinds[:] .= [diag_0_indices[row],diag_m1_indices[row],
    #                     diag_mnx_indices[row],diag_p1_indices[row],
    #                     diag_pnx_indices[row]]
    #     jac[arrayinds] .= [-A*(8*coupling+V[i,j]*(Pmat[i-1,j]+Pmat[i,j-1]-4*Pmat[i,j]+Pmat[i,j+1]+Pmat[i+1,j])-Vxshift[i,j]*(Pmat[i-1,j]-Pmat[i,j])+Vxshift[i+1,j]*(Pmat[i,j]-Pmat[i+1,j])-Vyshift[i,j]*(Pmat[i,j-1]-Pmat[i,j])+Vyshift[i,j+1]*(Pmat[i,j]-Pmat[i,j+1]))/(2*dx*dy),A*(2*coupling-V[i,j]*(Pmat[i-1,j]-Pmat[i,j])+Vxshift[i,j]*(Pmat[i-1,j]-Pmat[i,j]))/(2*dx*dy),A*(2*coupling-V[i,j]*(Pmat[i,j-1]-Pmat[i,j])+Vyshift[i,j]*(Pmat[i,j-1]-Pmat[i,j]))/(2*dx*dy),
    #                               A*(2*coupling+V[i,j]*(Pmat[i,j]-Pmat[i+1,j])-Vxshift[i+1,j]*(Pmat[i,j]-Pmat[i+1,j]))/(2*dx*dy),A*(2*coupling+V[i,j]*(Pmat[i,j]-Pmat[i,j+1])-Vyshift[i,j+1]*(Pmat[i,j]-Pmat[i,j+1]))/(2*dx*dy)]
    # end
    # jac[diagind(jac,nx-1)[1:nx:end-nx]] .= jac[diagind(jac,-1)[nx:nx:end]]
    # jac[diagind(jac,-nx+1)[1:nx:end-nx]] .= jac[diagind(jac,1)[nx:nx:end]]
    # jac[diagind(jac,-1)[nx:nx:end]] = 0
    # jac[diagind(jac,1)[nx:nx:end]] = 0
    # jac[diagind(jac,nx*ny-1)] = 0
    # jac[diagind(jac,-nx*ny+1)] = 0
    # jac[diagind(jac,nx*ny-nx)] = 0
    # jac[diagind(jac,-nx*ny+nx)] = 0
    jac
end

temperature_flux!(::Type{Val{:jac}},jac,P,params,t) = temperature_flux!(Val{:jac},jac,P,params...)
