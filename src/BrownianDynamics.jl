module BrownianDynamics

using ForwardDiff
using DifferentialEquations
using ToeplitzMatrices
using OffsetArrays
using Parameters

export # Types.
       SpectralParameters,
       # Functions.
       flux!,
       create_params,
       create_initial_conditions,
       get_variables,
       density_current,
       density_flux!,
       evolve_to_steady_state!,
       solve_steady_state,
       spectral_rhs!,
       conv,
       conv_mat,
       real_current


include("finitevolumes.jl")
include("spectral1D.jl")
include("util.jl")

end # module
