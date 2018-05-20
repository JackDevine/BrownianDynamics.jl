module BrownianDynamics

using ForwardDiff
using DifferentialEquations
using ToeplitzMatrices
using Parameters

export # Types.
       SpectralParameters,
       # Functions.
       flux!,
       create_params,
       create_initial_conditions,
       get_variables,
       density_current,
       find_steady!,
       spectral_rhs!,
       conv,
       conv_mat


include("finitevolumes.jl")
include("spectral1D.jl")
include("util.jl")

end # module
