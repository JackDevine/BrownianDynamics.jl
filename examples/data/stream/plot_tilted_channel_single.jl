include("stream.jl")
using JLD2
using Parameters
using LaTeXStrings
using OffsetArrays
using BrownianDynamics
using PyPlot
using SparseArrays
rc("text",usetex=true)
rc("font",size=10)
rc("contour",negative_linestyle="solid")
## Read in data.
@load "$(@__DIR__)/../steady_state_single_channel.jld" u_steady params xx yy T0  
#V_equilibrium -= minimum(V_equilibrium)

@unpack Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling = params

V .-= minimum(V)

# V_x = rotl90(V_x)
# V_y = rotl90(V_y)
nx,ny = size(V_x)[1]-1,size(V_x)[2]-1
T = reshape(u_steady[(nx*ny+1:end)],nx,ny)
P = reshape(u_steady[1:nx*ny],nx,ny)
Tsol = fill(T0,0:nx+1,0:ny+1)
Tsol[1:nx,1:ny] .= reshape(u_steady[(nx*ny+1):end],nx,ny)
# Periodicity in the x direction.
Tsol[0,1:ny] .= Tsol[nx-1,1:ny]
Tsol[nx+1,1:ny] .= Tsol[2,1:ny]

Psol = fill(0.0,0:nx+1,0:ny+1)
Psol[1:nx,1:ny] .= reshape(u_steady[1:nx*ny],nx,ny)
# Periodicity in the x direction.
Psol[0,1:ny] .= Psol[nx-1,1:ny]
Psol[nx+1,1:ny] .= Psol[2,1:ny]


Jx = [-( 0.5*(Psol[i-1,j]+Psol[i,j])*V_x[i,j]
        +0.5*(Tsol[i-1,j]+Tsol[i,j])*(Psol[i,j]-Psol[i-1,j])/dx) for i in 1:nx, j in 1:ny]
Jy = [-( 0.5*(Psol[i,j-1]+Psol[i,j])*V_y[i,j]
        +0.5*(Tsol[i,j-1]+Tsol[i,j])*(Psol[i,j]-Psol[i,j-1])/dy) for i in 1:nx, j in 1:ny]

Jx = rotl90(Jx)
Jy = -rotl90(Jy)
nablaT_x = [(Tsol[i,j]-Tsol[i-1,j])/dx for i in 1:nx, j in 1:ny]
nablaT_y = [(Tsol[i,j]-Tsol[i,j-1])/dy for i in 1:nx, j in 1:ny]
nablaT_x = rotl90(nablaT_x)
nablaT_y = -rotl90(nablaT_y)
Psol = rotl90(Psol)
Tsol = rotl90(Tsol)
P = rotl90(P)
# Plotting
close("all")
## Potential.
# figure(1,figsize=(3,12))
subplots(sharex=true,figsize=(3.1,9))
subplots_adjust(wspace=1)
p1 = subplot(4,1,1)
# Create a dummy contourf plot, so that we can steal its colorbar.
figure(2)
#V_equilibrium = rotl90(V_equilibrium)
#cbar_ax = contourf(V_equilibrium)
V = rotl90(V) # presume it is these we want rather thatn V_equilibrium - MJ
cbar_ax = contourf(V)


figure(1)
contour(xx,yy,V,linewidths=0.8,levels=range(minimum(V),stop=maximum(V),length=6))
colorbar(cbar_ax)
# colorbar(ticks=linspace(-2.5,18,5))
clim(-2.5,18)
ylabel(L"y/l")
xticks([])

## Probability distribution.
subplot(4,1,2)
nlevels = 6

P .-= minimum(P)
p3 = contour(xx,yy,P,levels=range(0.001,stop=maximum(P),length=nlevels),linewidths=0.8)
# Make an invisible colorbar, so that everything lines up.
figure(2)
cbar_ax = contourf(P) # want colour bar. Copy T example - MJ
figure(1)
colorbar(cbar_ax,ticks=round.(range(minimum(P),stop=maximum(P),length=4),digits=2))
#cbar_ax = colorbar()
#cbar_ax[:ax][:set_visible](false)
ylabel(L"y/l")
xticks([])

## Probability current.
p4 = subplot(4,1,3)
stride = 15
# Jx,Jy = density_current((xx,yy),u_steady,params)
# Jx = rotl90(Jx)
# Jy = rotl90(Jy)
ψ = flowfun(Jx,Jy)
contour(xx,yy,ψ,range(minimum(ψ),stop=maximum(ψ),length=6),colors="g",linewidths=0.8)
#contour(xx,yy,ψ,linspace(0,maximum(ψ),3),colors="g",linewidths=0.8)
# Make an invisible colorbar, so that everything lines up.
cbar_ax = colorbar()
cbar_ax.ax.set_visible(false)
quiver(xx[1:stride:end-1],yy[1:stride:end-1],Jx[1:stride:end-1,1:stride:end-1],
    Jy[1:stride:end-1,1:stride:end-1],scale=1.5,width=0.005,headwidth=4,zorder=2)
xticks([])
ylabel(L"y/l")
## Temperature and heat.
# figure(2,figsize=(1.5,1.5))
p2 = subplot(4,1,4)
stride = 25
T = rotl90(T)
temperature_contour = contour(xx,yy,T,linewidths=0.8)
# Create a dummy contourf plot, so that we can steal its colorbar.
figure(2)
cbar_ax = contourf(T)
figure(1)
colorbar(cbar_ax,ticks=round.(range(minimum(T),stop=maximum(T),length=4),digits=2))
quiver(xx[1:stride:end],yy[1:stride:end],-nablaT_x[1:stride:end,1:stride:end],
    -nablaT_y[1:stride:end,1:stride:end],zorder=2,width=0.005,headwidth=5,scale=1)
ylabel(L"y/l")
xlabel(L"x/L")
## Save the plot into the TeX directory.
savefig("$(@__DIR__)/../figures/single_channel.pdf",bbox_inches="tight")