using BrownianDynamics
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

@testset "flux!" begin
    # Test that the Jacobian is correct.
end
