#=
Calculate the steady state for multiple different parameters. Use the one dimensional
spectral methods.
=#
using DifferentialEquations
using BrownianDynamics
using OffsetArrays
## Initialize the system
nn = 128
p = OffsetArray(Complex{Float64},-round(Int,nn/2):round(Int,nn/2-1))
p[:] = zeros(Complex{Float64},nn)
p[-2:2] = [0.0 0.0 1.0 0.0 0.0]
vv = OffsetArray(Complex{Float64},-round(Int,nn/2):round(Int,nn/2-1))
vv[:] = 0.0
# vv[-2:2] = [0.25 0.0 1.0 0.0 0.25]
vv[-2:2] = [0.25 0.25im 1.0 -0.25im 0.25]
tt = OffsetArray(Complex{Float64},-round(Int, nn/2):round(Int,nn/2-1))
tt[:] = 0.0
tt[-1:1] = [0.2 1.0 0.2]

p = collect(p)
vv = collect(vv)
tt = collect(tt)
ϕ0 = [p;tt]

T = Complex{Float64}
nu = length(vv)
nv = length(vv)
n = nu+nv-1
np2 = n > 1024 ? nextprod([2,3,5],n) : nextpow2(n)
upad = [vv;zeros(T,np2-nu)]
vpad = [vv;zeros(T,np2-nv)]

plan = plan_fft!(upad)
iplan = plan_ifft!(upad)
y = similar(upad)

A = 1.0
B = [1.0 1.0]
force = 1.0
T0 = 1.0

params = SpectralParameters{Float64}(plan,iplan,A,B |> vec,force,vv,T0)

## Initialize the ODE problem.
tspan = (0.0,1.0)
prob = ODEProblem(spectral_rhs!,ϕ0,tspan,params)

## Find the Boltzmann distribution.
V = fft(fftshift(vv))
# Calculate the Boltmann distribution in real space.
# When the force is zero, the steady state temperature is equal to T0.
boltz = exp.(-V/params.T0)
p_boltz = ifftshift(ifft(boltz))
# Normalization.
nd2 = round(Int,nn/2)
p_boltz /= p_boltz[nd2+1]
# Steady state temperature.
tt_equilibrium = fill(0.0+0.0im,nn)
tt_equilibrium[nd2+1] = T0

ϕ_equilibrium = [p_boltz;tt_equilibrium]
equilibrium_params = SpectralParameters{Float64}(plan,iplan,A,B |> vec,0,vv,T0)
# Check that we are in the steady state.
@assert norm(spectral_rhs!(copy(ϕ0),ϕ_equilibrium,equilibrium_params,0)) < 1e-8

## Calculate the steady state for multiple parameters.
nforces = 10
nAs = 10
forcevec = linspace(0,1.5,nforces)
Avec = linspace(1,3,nAs)
Jxsum = Array{Float64}(nforces,nAs)
errorvec = Array{Float64}(nforces,nAs)
solutions = Array{Complex{Float64}}(2nn,nforces,nAs)

# When the force is zero, we will have the equilibrium steady state.
params = SpectralParameters{Float64}(plan,iplan,A,B |> vec,forcevec[1],vv,T0)
prob = remake(prob,p=params,u0=copy(ϕ_equilibrium))
integrator = init(prob,TRBDF2(autodiff=false),abstol=1e-10,reltol=1e-10)
@time steady_state = evolve_to_steady_state!(integrator,params,steadytol=1e-3)

@time for (i,force) in enumerate(forcevec)
    for (j,A) in enumerate(Avec)
        # Redefine the problem for the current value of the force.
        params = SpectralParameters{Float64}(plan,iplan,A,B |> vec,force,vv,T0)
        prob = remake(prob,p=params,u0=steady_state)
        integrator = init(prob,TRBDF2(autodiff=false),abstol=1e-10,reltol=1e-10)
        # Find the steady state, first evolve the equation with the integrator.
        # Use the result as a starting point for Newton's method.
        steady_state[:] .= evolve_to_steady_state!(integrator,params,steadytol=1e-5,
                                                   maxiters=20,convergence_warning=false)
        steady_state[:] .= solve_steady_state(integrator,params,steadytol=1e-8,
                                                                maxiters=200)
        # Store results.
        solutions[:,i,j] .= steady_state
        Jxsum[i,j] = real_current(integrator.u,params) |> sum |> real
        errorvec[i,j] = norm([spectral_rhs!(copy(ϕ0),steady_state,params,0)
                              ;steady_state[nd2+1]-1])/nn
    end
    steady_state[:] .= solutions[:,i,1]
end
