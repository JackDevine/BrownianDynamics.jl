# BrownianDynamics
Source code for my masters thesis:
https://ourarchive.otago.ac.nz/handle/10523/8237

The code solves a set of coupled nonlinear partial differential equations using the finite volume discretization. There is also a spectral discretization in the file `src/spectral.jl`.

The `examples` folder contains some Jupyter notebooks showing some typical uses of the code.

## Installation
Simply open a Julia Repl and enter the Pkg repl mode,then paste the following:
```julia
add https://github.com/JackDevine/BrownianDynamics.jl
```
