module BrownianDynamics

using ForwardDiff
using ToeplitzMatrices
using OffsetArrays
using Parameters
using StaticArrays
using SymPy
using DSP
using FFTW
using LinearAlgebra
using SparseArrays

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
       density!,
       temperature!,
       conv,
       conv_mat,
       real_current,
       create_system


include("finitevolumes.jl")
include("symbolics.jl")
include("spectral1D.jl")
include("util.jl")

end # module
