#=
These functions are taken from http://pordlabs.ucsd.edu/matlab/stream.htm
Kirill K. Pankratov, March 7, 1994.
=#

function cumsimp(vec::Vector{T}) where T
    c1 = 3//8
    c2 = 6//8
    c3 = -1//8

    # Interpolate values of Y to all midpoints
    f = fill(zero(T),length(vec))
    for i in 2:length(f)-1
        f[i] = c1*vec[i-1]+c2*vec[i]+c3*vec[i+1]
    end
    for i in 3:length(f)
        f[i] += c3*vec[i-2]+c2*vec[i-1]+c1*vec[i]
    end
    f[2] *= 2
    f[end] *= 2
    # Now Simpson (1,4,1) rule
    for i in 2:length(f)
        f[i] = 2f[i]+vec[i-1]+vec[i]
    end
    f[1:end] .= cumsum(f)/6  # Cumulative sum, 6 - denom. from the Simpson rule
end

function cumsimp(U::AbstractMatrix{T}) where T
    ret = similar(U)
    for i in 1:size(U)[2]
        ret[:,i] .= cumsimp(U[:,i])
    end
    ret
end

function flowfun(u,v)
    lx = size(u,2)  # Size of the velocity matrices
    ly = size(u,1)
    # Integrate velocity fields to get potential and streamfunction
    # Use Simpson rule summation
    cx = cumsimp(v[1,:])  # Compute x-integration constant
    cy = cumsimp(u[:,1])  # Compute y-integration constant
    ψ = -cumsimp(u)+repeat(cx',ly,1)
    ψ = (ψ+cumsimp(v')'-repeat(cy,1,lx))/2
end

# function cumsimp_naive(vec::Vector{T})::Vector{T} where T
#     c1 = 3//8
#     c2 = 6//8
#     c3 = -1//8
#
#     lv = length(vec)
#     num = 1:lv-2
#     vec_itp = fill(zero(T),length(vec))
#     # Interpolate values of Y to all midpoints
#     @. vec_itp[num+1] = c1*vec[num]+c2*vec[num+1]+c3*vec[num+2]
#     @. vec_itp[num+2] = vec_itp[num+2]+c3*vec[num]+c2*vec[num+1]+c1*vec[num+2]
#     vec_itp[2] *= 2
#     vec_itp[lv] *= 2
#     # Now Simpson (1,4,1) rule
#     @. vec_itp[num+1] = 2vec_itp[num+1]+vec[num]+vec[num+1]
#     vec_itp[:] .= cumsum(vec_itp)/6
# end
