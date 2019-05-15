include("stream.jl")
# using Plots; pyplot()
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
rc("font",family="DejaVu Sans")
## Read in data.
cd(@__DIR__)
@load "$(pwd())/../steady_state_heat_engine.jld" u_steady params xx yy T0 V_equilibrium
# V_equilibrium -= minimum(V_equilibrium)
@unpack Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling = params
nx,ny = size(V_x)[1]-1,size(V_x)[2]-1
T = reshape(u_steady[(nx*ny+1:end)],nx,ny)
P = reshape(u_steady[1:nx*ny],nx,ny)
Tsol = fill(T0,0:nx+1,0:ny+1)
Tsol[1:nx,1:ny] .= reshape(u_steady[(nx*ny+1):end],nx,ny)
# Periodicity in the x direction.
Tsol[0,1:ny] .= Tsol[nx-1,1:ny]
Tsol[nx+1,1:ny] .= Tsol[2,1:ny]

nablaT_x = [(Tsol[i,j]-Tsol[i-1,j])/dx for i in 2:nx, j in 1:ny]
nablaT_y = [(Tsol[i,j]-Tsol[i,j-1])/dy for i in 2:nx, j in 1:ny]
Jx,Jy = density_current((xx,yy),u_steady,params)
Jx = rotr90(Jx) # fixed this so it is doing the right thing - MJ
Jy = -rotr90(Jy)
T = rotr90(T)
nablaT_x = -rotr90(nablaT_x)
nablaT_y = rotr90(nablaT_y)
# xx = reverse(xx)
# yy = reverse(yy)
# Plotting
## Potential.
close("all")
figure(1,figsize=(3,3))
contour(xx,yy,rotl90(V_equilibrium),linewidths=0.6)
hlines([1],0,1,linewidth=2,color="blue")
annotate("Cold bath: "*L"T=T_{C}",xy=(0.25,1.05))
hlines([-1],0,1,linewidth=2,color="red")
annotate("Hot bath: "*L"T=T_{H}",xy=(0.25,-1.18))
ylim(-1.2,1.2)
ylabel(L"y/l")
xlabel(L"x/L")
annotate("1",xy=(0.15,0.6))
annotate("2",xy=(0.5,0.0))
annotate("3",xy=(0.85,-0.7))
savefig("../figures/heat_engine_potential.pdf",bbox_inches="tight")
##
# Create a dummy contourf plot, so that we can steal its colorbar.
figure(3)
cbar_ax = contourf(V_equilibrium)
figure(1)
colorbar(cbar_ax)
# savefig("../../TwoDPaper/figures/heat_engine_potential.pdf",bbox_inches="tight")
## First do the heat engine.
## Probability current.
figure(2,figsize=(3.1,9))
plt = subplot(4,1,1)
subplots_adjust(hspace=0.22,wspace=0.08)
stride = 10
quiver(xx[3:stride:end-1],yy[3:stride:end-1],Jx[3:stride:end-1,2:stride:end-1],
       Jy[3:stride:end-1,2:stride:end-1],zorder=2,scale=6,width=0.005,headwidth=4)
ψ = flowfun(Jx[2:end-1,2:end-1],Jy[2:end-1,2:end-1])

#cs = contour(xx[2:end-1],yy[2:end-1],ψ,colors="g",
 #       levels=linspace(minimum(ψ),maximum(ψ),6), 
 #       linewidths=0.8,linestyle="solid")


cs = contour(xx[2:end-1],yy[2:end-1],ψ,colors="g",
        levels=[-1.842,0.526,2.805],
       linewidths=0.8,linestyle="solid")
# clabel(cs,inline=1)
##
# close("all")
# contour(ψ,levels=[1.0,5.0,8.3])
# colorbar()
##
# Make an invisible colorbar, so that everything lines up.
cbar_ax = colorbar()
cbar_ax.ax.set_visible(false)
ylabel(L"y/l")
xticks([])
# Quiver plot.
ylabel(L"y/l")
yticks([-1,-0.5,0,0.5,1])
xticks([])
## Temperature and heat.
subplot(4,1,2)
stride = 14
quiver(xx[2:stride:end],yy[2:stride:end],-nablaT_x[2:stride:end,1:stride:end],
       -nablaT_y[2:stride:end,1:stride:end],zorder=2,scale=25,width=0.005,headwidth=4)
contour(xx,yy,T,linewidths=1,colorbar=true,zorder=2)
# Create a dummy contourf plot, so that we can steal its colorbar.
figure(3)
cbar_ax = contourf(T)
figure(2)
colorbar(cbar_ax,ticks=round.(range(1,stop=3,length=4),digits=2))
xticks([])
# yticks([-1,0])
ylabel(L"y/l")
## Now the heat pump.
@load "$(pwd())/../steady_state_heat_pump.jld" u_steady params xx yy T0 V_equilibrium
@unpack Pmat,Tmat,Jx,Jy,V,Vxshift,Vyshift,V_x,V_y,dx,dy,A,coupling = params
nx,ny = size(V_x)[1]-1,size(V_x)[2]-1
T = reshape(u_steady[(nx*ny+1:end)],nx,ny)
P = reshape(u_steady[1:nx*ny],nx,ny)
Tsol = fill(T0,0:nx+1,0:ny+1)
Tsol[1:nx,1:ny] .= reshape(u_steady[(nx*ny+1):end],nx,ny)
# Periodicity in the x direction.
Tsol[0,1:ny] .= Tsol[nx-1,1:ny]
Tsol[nx+1,1:ny] .= Tsol[2,1:ny]


nablaT_x = [(Tsol[i,j]-Tsol[i-1,j])/dx for i in 1:nx, j in 1:ny]
nablaT_y = [(Tsol[i,j]-Tsol[i,j-1])/dy for i in 1:nx, j in 1:ny]
Jx,Jy = density_current((xx,yy),u_steady,params)
Jx = rotr90(Jx)
Jy = -rotr90(Jy)
T = rotr90(T)
nablaT_x = -rotr90(nablaT_x)
nablaT_y = rotr90(nablaT_y)
# xx = reverse(xx)
# yy = reverse(yy)
# @load "$(pwd())/../steady_state_heat_pump2018-07-03.jld" u_steady params xx yy T0 V_equilibrium
# nx,ny = length(xx),length(yy)
# T = Tmat[1:nx,1:ny]
# nablaT_x = ∇Tx
# nablaT_y = ∇Ty
# T = rotr90(T)
# nablaT_x = -rotr90(nablaT_x)
# nablaT_y = rotr90(nablaT_y)
# Jx = -rotr90(Jx)
# Jy = rotr90(Jy)
## Probability current.
subplot(4,1,3)
ψ = flowfun(Jx[2:end-1,2:end-1],Jy[2:end-1,2:end-1])
contour(xx[2:end-1],yy[2:end-1],ψ,colors="g",levels=range(minimum(ψ),stop=maximum(ψ),length=6),
        linewidths=0.6,linestyle="solid")
# Make an invisible colorbar, so that everything lines up.
cbar_ax = colorbar()
cbar_ax.ax.set_visible(false)
ylabel(L"y/l")
# Quiver plot.
stride = 10
quiver(xx[3:stride:end-1],yy[3:stride:end-1],Jx[3:stride:end-1,2:stride:end-1],
    Jy[3:stride:end-1,2:stride:end-1],zorder=2,scale=0.5,width=0.004,headwidth=4)
xticks([])
# xlabel(L"x")
ylabel(L"y/l")
yticks([-1,-0.5,0,0.5,1])
## Temperature and heat.
subplot(4,1,4)
stride = 14
quiver(xx[2:stride:end],yy[2:stride:end],-nablaT_x[2:stride:end,2:stride:end],
       -nablaT_y[2:stride:end,2:stride:end],zorder=2,scale=0.5,width=0.005,headwidth=4)
contour(xx,yy,T,linewidths=1,colorbar=true,zorder=2,vmin=1.00)
# Create a dummy contourf plot, so that we can steal its colorbar.
figure(3)
# cbar_ax = contourf(T)
cbar_ax = contourf(xx,yy,T,range(1.00,stop=1.02,length=6))
figure(2)
colorbar(cbar_ax,ticks=range(1.00,stop=1.02,length=3))
# yticks([-1,0])
xlabel(L"x/L")
ylabel(L"y/l")
## Save the plot into the TeX directory.
savefig("../figures/heat_engine.pdf",bbox_inches="tight")
