using BrownianDynamics
using OffsetArrays
using DifferentialEquations
using Base.Test

# write your own tests here
@testset "conv_mat" begin
    vv = [1.0;1.0+im;2.0;1.0*im;1.0;3.0+im]
    T = Complex{Float64}
    nu = length(vv)

    n = 2nu-1
    np2 = n > 1024 ? nextprod([2,3,5],n) : nextpow2(n)
    upad = [vv;zeros(T,np2-nu)]

    p = plan_fft!(upad)
    ip = plan_ifft!(upad)

    @test conv_mat(vv) == conv_mat(p,ip,vv)
    @test isapprox(conv_mat(p,ip,vv)*vv,conv(vv,vv)[4:9])
    u = rand(6)
    v = rand(6)
    @test isapprox(conv_mat(u)*v,conv(u,v)[4:9])
end

# @testset "flux!" begin
#     # Test that the Jacobian is correct.
# end

## Initialize a spectral system.
nn = 64
nd2 = round(Int,nn/2)
p = OffsetArray(Complex{Float64},-round(Int,nn/2):round(Int,nn/2-1))
p[:] = zeros(Complex{Float64},nn)
p[-2:2] = [0.0 0.0 1.0 0.0 0.0]
vv = OffsetArray(Complex{Float64},-round(Int,nn/2):round(Int,nn/2-1))
vv[:] = 0.0
# vv[-2:2] = [0.25 0.0 1.0 0.0 0.25]
vv[-2:2] = 2*[0.0 0.5im 0.0 -0.5im 0.0]
tt = OffsetArray(Complex{Float64},-round(Int, nn/2):round(Int,nn/2-1))
tt[:] = 0.0
tt[-1:1] = [0.0 1.0 0.0]

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

A = 4.0
B = [1.0 100.0]
force = 4.0
T0 = 1.0

params = SpectralParameters{Float64}(plan,iplan,A,B |> vec,force,vv,T0)
tspan = (0.0,0.5)
prob = ODEProblem(spectral_rhs!,ϕ0,tspan,params)

@testset "spectral_rhs!" begin
    jac_diff = DiffEqDiffTools.finite_difference_jacobian((dϕ,ϕ) -> spectral_rhs!(dϕ,ϕ,params,0),ϕ0)
    jac = spzeros(Complex{Float64},2nn,2nn)
    jac = spectral_rhs!(Val{:jac},jac,ϕ0,params,0)
    @test norm(jac*ϕ0 - jac_diff*ϕ0)/nn < 1e-9

    sol = solve(prob,TRBDF2(autodiff=false),abstol=1e-10,reltol=1e-10)

    jac_diff = DiffEqDiffTools.finite_difference_jacobian((dϕ,ϕ) -> spectral_rhs!(dϕ,ϕ,params,0),sol[end])
    jac = spzeros(Complex{Float64},2nn,2nn)
    jac = spectral_rhs!(Val{:jac},jac,sol[end],params,0)
    @test norm(jac*sol[end] - jac_diff*sol[end])/nn < 1e-9
end

@testset "solve_steady_state" begin
    integrator = init(prob,TRBDF2(autodiff=false))
    u = solve_steady_state(integrator,params,steadytol=1e-10,maxiters=5)
    @test norm([spectral_rhs!(copy(ϕ0),u,params,0);u[nd2+1]-1])/nn < 1e-10
end
