function create_params(mesh,potential,A,coupling,T0)
    xx,yy = mesh
    nx = length(xx)
    ny = length(yy)
    dx,dy = (xx[end]-xx[1])/nx,(yy[end]-yy[1])/ny

    # Discrete potential. We will need the potential at the center of the volumes as well as the edges.
    V = [potential(x,y) for y in yy, x in xx]
    V_x = [ForwardDiff.derivative(xp->potential(xp,y),x-0.5dx) for y in yy, x in xx]
    V_y = [ForwardDiff.derivative(yp->potential(x,yp),y-0.5dy) for y in yy, x in xx]
    Vxshift = [potential(x-0.5dx,y) for y in [yy;yy[end]+dy], x in [xx;xx[end]+dx]]
    Vyshift = [potential(x,y-0.5dy) for y in [yy;yy[end]+dy], x in [xx;xx[end]+dx]]

    params = (A,coupling,T0,V,Vxshift,Vyshift,V_x,V_y,dx,dy)
end

function create_initial_conditions(mesh,density_init,temperature_init)
    xx,yy = mesh
    nx = length(xx)
    ny = length(yy)
    dx,dy = (xx[end]-xx[1])/nx,(yy[end]-yy[1])/ny
    # Discrete versions of the temperature and the probability.
    P = [density_init(x,y) for y in yy, x in xx]
    P /= sum(P)*dx*dy  # Normalize probability.
    T = [temperature_init(x,y) for y in yy, x in xx]

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
    (iters >= maxiters && convergence_warning) && println("Maximum number of iterations reached, exiting.")

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
    while (norm([du;u[nd2+1]-1])/nn>steadytol) && (iters<maxiters)
        du[:] .= integrator.f(du,u,params,0)
        jac[:,:] .= integrator.f(Val{:jac},jac,u,params,0)

        u[:] .+= -(([jac boundary_jac;boundary_jac' 0.0])\[du;u[nd2+1]-1])[1:2nn]
        iters += 1
    end
    iters >= maxiters && println("Maximum number of iterations reached, exiting.")

    u
end
