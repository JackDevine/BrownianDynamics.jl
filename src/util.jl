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
    Pmat = OffsetArray(eltype(xx),0:nx+1,0:ny+1)
    Pmat[0:nx+1,0] = 0
    Pmat[0:nx+1,ny+1] = 0
    Tmat = OffsetArray(eltype(xx),0:nx+1,0:ny+1)
    Tmat[0:nx+1,0:ny+1] = [temperature_init(x,y) for x in [xx[1]-dx;xx;xx[end]+dx],
                                                     y in [yy[1]-dy;yy;yy[end]+dy]]
    Jx = Array{eltype(xx)}(nx+1,ny+1)
    Jy = Array{eltype(xx)}(nx+1,ny+1)

    params = (Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling)
end

function create_initial_conditions(mesh,density_init,temperature_init)
    xx,yy = mesh
    nx = length(xx)
    ny = length(yy)
    # dx,dy = (xx[end]-xx[1])/nx,(yy[end]-yy[1])/ny
    dx,dy = xx[2]-xx[1],yy[2]-yy[1]
    # Discrete versions of the temperature and the probability.
    P = [density_init(x,y) for x in xx, y in yy]
    P /= sum(P)*dx*dy  # Normalize probability.
    T = [temperature_init(x,y) for x in xx, y in yy]

    u0 = [P[:];T[:]]
end

function evolve_to_steady_state!(integrator,params;steadytol=1e-3,maxiters=500,
                                                   convergence_warning=true)
    du = Array{eltype(integrator.u)}(size(integrator.u))
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
    jac = Array{eltype(u)}(length(u),length(u))
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

    u,residual_list
end

# TODO sure that we can find out whether the discretisation is FVM via dispatch.
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

function solve_steady_state(integrator,params::Tuple;
                            steadytol=1e-3,maxiters=10,print_residual=true)
    Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling = params
    u = copy(integrator.u)
    nn = round(Int,length(u)/2)
    nx = round(Int,sqrt(nn))

    du = similar(u)
    # We will use the boundary jacobian to assert that `sum(u[1:nx*ny])*dx*dy == 1`.
    boundary_jac = zeros(2nn)
    boundary_jac[1:nn] = one(eltype(u))
    jac = spzeros(2nn,2nn)
    #=
    Use this if you want to autodiff the Jacobian.
    # du[:] .= integrator.f(du,u,params,0)
    # params_autodiff = params[5:end]
    # f_autodiff = (du,u) -> flux_autodiff!(du,u,params_autodiff,0)
    # jac[:,:] .= ForwardDiff.jacobian!(jac,f_autodiff,du,u)
    # fill!(jac,zero(eltype(jac)))
    =#
    # Newton's method.
    du[:] .= integrator.f(du,u,params,0)
    iters = 0
    while (norm([du;sum(u[1:nn])*dx*dy-1])/nn>steadytol) && (iters<maxiters)
        # jac[:,:] .= ForwardDiff.jacobian!(jac,f_autodiff,du,u)
        jac[:,:] .= flux!(Val{:jac},jac,u,params,0)

        du[:] .= integrator.f(du,u,params,0)
        u[:] .+= -(([jac boundary_jac;boundary_jac' 0.0])\[du;sum(u[1:nn])*dx*dy-1])[1:2nn]
        iters += 1
    end
    iters > maxiters && println("Maximum number of iterations reached, exiting.")
    print_residual && @show norm([du;sum(u[1:nn])*dx*dy-1])/nn

    u
end

function stencil_indices!(inds,dims,row)
    n1,n2 = dims
    inds[:] .= SVector(row,row-1,row-n1,row+1,row+n2)-one(eltype(inds))
    # inds -= one(eltype(inds))
    i,j = ind2sub((n1,n2),row)
    if i==1
        inds[2] += n1-one(eltype(inds))
    elseif i==n1
        inds[4] -= n1-one(eltype(inds))
    end
    for i in 1:length(inds)
        inds[i] = mod(inds[i],n1*n2)+one(eltype(inds))
    end
    # inds = @SVector [mod(index,n1*n2)+one(eltype(inds)) for index in inds]
    inds
end
