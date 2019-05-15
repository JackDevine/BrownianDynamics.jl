"""
    create_system(xx,yy,Tc,Th,coupling,
                  p;potential_type=:tilted_channel) -> (u0,params)

Initializes a system to be solved by the finite volume solvers in
BrownianDynamics. `u0` is an initial condition for the system and `params`
is the parameters of the system of type `FVMParameters`.
"""
function create_system(xx,yy,Tc,Th,coupling,p;potential_type=:tilted_channel)
    if potential_type == :tilted_channel
        potential = (x,y) -> -p[1]*x+p[2]cos(2π*x)+p[3]*y^2
    elseif potential_type == :curved_channel
        potential = (x,y) -> (-p[1]*x-p[2]*exp(-0.5*(-sin(2π*x)-sin(2π*(0.5y+0.5))+2cos(π*(0.5y+0.5)))^2)
                              +p[3]*(y)^4-p[4]*cos(2π*x)*cos(2π*(0.5y+0.5)))
    elseif potential_type == :double_channel
        potential = (x,y) -> ( exp(-10*(y+0.5)^2)*(-p[4]+p[2]*cos(2π*(x+π/4)))
                              +exp(-10*(y-0.5)^2)*(-p[5]+p[3]*cos(2π*(x-π/4)))
                              -p[1]*x+p[6]*y^2)
    end

    A = 1.0
    density_init = (x,y) -> exp(-potential(x,y))
    temperature_init = (x,y) -> ((Tc-Th)/(yy[end]-yy[1]))*y+Th-((Tc-Th)/(yy[end]-yy[1]))*yy[1]

    params = create_params((xx,yy),potential,A,coupling,temperature_init)
    u0 = create_initial_conditions((xx,yy),density_init,temperature_init)

    u0,params
end

function create_params(mesh,potential,A,coupling,temperature_init)
    xx,yy = mesh
    nx = length(xx)
    ny = length(yy)
    dx,dy = xx[2]-xx[1],yy[2]-yy[1]
    # Discrete potential. We will need the potential at the center of the volumes, as well
    # as the edges. The gradient of the potential only needs to be evaluated at the edges
    # of the cells.
    V = [potential(x,y) for x in xx, y in yy]
    V_x = [ForwardDiff.derivative(xp->potential(xp,y),x-0.5dx) for x in [xx;xx[end]+dx], y in [yy;yy[end]+dy]]
    V_y = [ForwardDiff.derivative(yp->potential(x,yp),y-0.5dy) for x in [xx;xx[end]+dx], y in [yy;yy[end]+dy]]
    Vxshift = [potential(x-0.5dx,y) for x in [xx;xx[end]+dx], y in [yy;yy[end]+dy]]
    Vyshift = [potential(x,y-0.5dy) for x in [xx;xx[end]+dx], y in [yy;yy[end]+dy]]
    Pmat = OffsetArray{eltype(xx)}(undef,0:nx+1,0:ny+1)
    fill!(Pmat,zero(eltype(xx)))
    Pmat[0:nx+1,0] *= 0
    Pmat[0:nx+1,ny+1] *= 0
    Tmat = OffsetArray{eltype(xx)}(undef,0:nx+1,0:ny+1)
    fill!(Tmat,zero(eltype(xx)))
    Tmat[0:nx+1,0:ny+1] = [temperature_init(x,y) for x in [xx[1]-dx;xx;xx[end]+dx],
                                                     y in [yy[1]-dy;yy;yy[end]+dy]]
    Jx = Array{eltype(xx)}(undef,nx+1,ny+1)
    Jy = Array{eltype(xx)}(undef,nx+1,ny+1)

    # jac = spzeros(eltype(xx),2nx*ny,2nx*ny)
    # jac_prototype = spzeros(eltype(xx),nx*ny,nx*ny)
    # jac_prototype[diagind(jac_prototype,0)] .= one(eltype(xx))*eps()
    # jac_prototype[diagind(jac_prototype,1)] .= one(eltype(xx))*eps()
    # jac_prototype[diagind(jac_prototype,-1)] .= one(eltype(xx))*eps()
    # jac_prototype[diagind(jac_prototype,nx*ny-1)] .= one(eltype(xx))*eps()
    # jac_prototype[diagind(jac_prototype,-nx*ny+1)] .= one(eltype(xx))*eps()
    # jac_prototype[diagind(jac_prototype,nx*ny-nx)] .= one(eltype(xx))*eps()
    # jac_prototype[diagind(jac_prototype,-nx*ny+nx)] .= one(eltype(xx))*eps()
    # jac_prototype[diagind(jac_prototype,nx*ny-1)] .= one(eltype(xx))*eps()
    # jac_prototype[diagind(jac_prototype,-nx*ny+1)] .= one(eltype(xx))*eps()
    # jac_prototype[diagind(jac_prototype,nx*ny-nx)] .= one(eltype(xx))*eps()
    # jac_prototype[diagind(jac_prototype,-nx*ny+nx)] .= one(eltype(xx))*eps()
    jac_prototype = spdiagm(0 => ones(eltype(xx),nx*ny),
                            1 => ones(eltype(xx),nx*ny-1),
                            -1 => ones(eltype(xx),nx*ny-1),
                            nx => ones(eltype(xx),nx*ny-nx),
                            -nx => ones(eltype(xx),nx*ny-nx),
                            nx*ny-nx => ones(eltype(xx),nx),
                            -nx*ny+nx => ones(eltype(xx),nx),
                            nx*ny-1 => ones(eltype(xx),1),
                            -nx*ny+1 => ones(eltype(xx),1))
    jac = vcat(hcat(jac_prototype,jac_prototype),hcat(jac_prototype,jac_prototype))

    FVMParameters(Pmat,Tmat,jac,Jx,Jy,V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling)
end

function create_initial_conditions(mesh,density_init,temperature_init)
    xx,yy = mesh
    nx = length(xx)
    ny = length(yy)
    dx,dy = xx[2]-xx[1],yy[2]-yy[1]
    # Discrete versions of the temperature and the probability.
    P = [density_init(x,y) for x in xx, y in yy]
    P /= sum(P)*dx*dy  # Normalize probability.
    T = [temperature_init(x,y) for x in xx, y in yy]

    u0 = [P[:];T[:]]
end

function evolve_to_steady_state!(integrator,params;steadytol=1e-3,maxiters=500,
                                                   convergence_warning=true)
    du = Array{eltype(integrator.u)}(undef,size(integrator.u))
    iters = 0
    # TODO right now du will be calculated somewhere in the step! function.
    # I then recalculate du, which is a bit of a waste.
    while ((norm(integrator.f(du,integrator.u,params,0))/size(integrator.u)[1] > steadytol)
            && (iters < maxiters))
        step!(integrator)
        iters += 1
    end
    (iters > maxiters && convergence_warning) && println("Maximum number of iterations reached, exiting.")

    integrator.u
end

function solve_steady_state(integrator,params::SpectralParameters{T};
                            steadytol=1e-3,maxiters=50) where T <: Number
    u = copy(integrator.u)
    nn = round(Int,length(u)/2)
    nd2 = round(Int,nn/2)
    du = similar(u)
    # We will use the boundary jacobian to assert that `u[nd2+1] == 1`.
    boundary_jac = zeros(eltype(u),2nn)
    boundary_jac[nd2+1] = 1
    jac = Array{eltype(u)}(undef,length(u),length(u))
    du[:] .= integrator.f(du,u,params,0)
    jac[:,:] .= integrator.f(Val{:jac},jac,u,params,0)
    # Newton's method.
    iters = 0
    residual_list = Float64[]
    while (norm([du;u[nd2+1]-1])/nn>steadytol) && (iters<maxiters)
        du[:] .= integrator.f(du,u,params,0)
        push!(residual_list,norm(du)/nn)
        jac[:,:] .= integrator.f(Val{:jac},jac,u,params,0)

        u[:] .+= -(([jac boundary_jac;boundary_jac' 0.0])\[du;u[nd2+1]-1])[1:2nn]
        iters += 1
    end
    iters > maxiters && println("Maximum number of iterations reached, exiting.")

    u
end

function solve_steady_state_uncoupled(integrator,params;
                                      steadytol=1e-3,maxiters=10,print_residual=true)
    Pmat,Tmat,Jx,Jy,V_x,V_y,dx,dy = params
    u = copy(integrator.u)
    nn = round(Int,length(u))

    du = similar(u)
    # We will use the boundary jacobian to assert that `sum(u[1:nx*ny])*dx*dy == 1`.
    boundary_jac = ones(nn)
    jac = spzeros(nn,nn)
    du[:] .= integrator.f(du,u,params,0)
    jac[:,:] .= integrator.f(Val{:jac},jac,u,params,0)
    # Newton's method.
    iters = 0
    while (norm([du;sum(u[1:nn])*dx*dy-1])/nn>steadytol) && (iters<maxiters)
        jac[:,:] .= integrator.f(Val{:jac},jac,u,params,0)

        du[:] .= integrator.f(du,u,params,0)
        u[:] .+= -(([jac boundary_jac;boundary_jac' 0.0])\[du;sum(u)*dx*dy-1])[1:nx*ny]
        iters += 1
    end
    iters > maxiters && println("Maximum number of iterations reached, exiting.")
    print_residual && @show norm([du;sum(u[1:nn])*dx*dy-1])/nn

    u
end

function solve_steady_state(u_init,params::FVMParameters;steadytol::Float64=1e-10,
                            maxiters::Int=5,print_residual::Bool=true,autodiff=false)
    dx,dy = params.dx,params.dy
    jac = params.jac_
    u = copy(u_init)
    nn = round(Int,length(u)/2)
    nx = round(Int,sqrt(nn))

    du = similar(u)
    # We will use the boundary jacobian to assert that `sum(u[1:nx*ny])*dx*dy == 1`.
    boundary_jac = zeros(eltype(u),2nn)
    boundary_jac[1:nn] .= one(eltype(u))
    du[:] .= flux!(du,u,params,0)
    # Autodiff code.
    if autodiff
        @unpack Pmat,Tmat,jac_,Jx,Jy,V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling = params
        params_autodiff = (V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling)
        f_autodiff = (du,u) -> flux_autodiff!(du,u,params_autodiff,0)
        tmp = ForwardDiff.jacobian!(jac,f_autodiff,du,u)
    end
    # Newton's method.
    du[:] .= flux!(du,u,params,0)
    iters = zero(Int64)
    while (norm([du;sum(u[1:nn])*dx*dy-1])/nn>steadytol) && (iters<maxiters)
        if autodiff
            jac[:,:] .= ForwardDiff.jacobian!(jac,f_autodiff,du,u)
            flux_autodiff!(du,u,params_autodiff,0)
        else
            flux!(Val{:jac},jac,u,params,0)
            flux!(du,u,params,0)
        end

        u[:] .+= -(([jac boundary_jac;boundary_jac' zero(eltype(u))])\[du;sum(u[1:nn])*dx*dy-one(eltype(u))])[1:2nn]
        iters += one(Int64)
    end
    iters > maxiters && println("Maximum number of iterations reached, exiting.")
    print_residual && @show norm([du;sum(u[1:nn])*dx*dy-1])/nn

    u
end

function stencil_indices!(inds,dims,row)
    n1,n2 = dims
    inds[:] .= SVector(row,row-1,row-n1,row+1,row+n2).-one(eltype(inds))
    i2s = CartesianIndices((n1,n2))
    i_j = i2s[row]
    i,j = i_j[1],i_j[2]
    if i==1
        inds[2] += n1-one(eltype(inds))
    elseif i==n1
        inds[4] -= n1-one(eltype(inds))
    end
    for i in 1:length(inds)
        inds[i] = mod(inds[i],n1*n2)+one(eltype(inds))
    end
    inds
end
