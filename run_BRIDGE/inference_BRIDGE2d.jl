# Import packages
using Optim, Statistics, Distributions, Plots,StatsBase,DelimitedFiles,Random
using FastGaussQuadrature,Flux,DataFrames,CSV,HypergeometricFunctions
include("../utils.jl")

# Define Gaussian Quadrature points and corresponding weights
n = 7
a,b = [0,1]
interval_X, weights = gausslegendre(n)
x = ((b - a) .* interval_X .+ b .+ a) ./ 2
w = weights * (b - a) / 2

z1 = x
z2 = x

W = vec(w*w')

# Define hidden channels
hidden_channels = 40

# Initialize BRIGE model
model = Chain(Dense(length(z1)+1, hidden_channels,tanh),Dense(hidden_channels, length(z1)*length(z2)),x -> softplus.(x))
params, re = Flux.destructure(model);
ps = Flux.params(params);

# Read trained parameters
using CSV,DataFrames
df = CSV.read("parameters_trained/params_trained2d.txt",DataFrame)
params = df.params
ps = Flux.params(params);

# Read inference counts data
# True value is [σ_on,σ_off,ρ,d] =  [0.595,7.335,37.865,0.474]. You can replace it with your own data.
SSA_counts = readdlm("dataset/synthetic_data/counts_example2d.txt")
U_sample = Int.(SSA_counts[:,1])
S_sample = Int.(SSA_counts[:,2])
Sample_size = length(U_sample)

# Convert counts data to joint distribution
US_sample = [[U_sample[i],S_sample[i]] for i=1:Sample_size]
U_max = Int(maximum([n for (n, m) in US_sample]))
S_max = Int(maximum([m for (n, m) in US_sample]))

joint_prob_matrix = zeros(Float64, U_max+1, S_max+1)
for (m, n) in US_sample
    joint_prob_matrix[m+1, n+1] += 1
end

joint_prob_matrix /= length(US_sample)

# Convert joint distribution to PGF
SSA_PGF = vec(hist_gf2d(joint_prob_matrix,z1,z2)')

# Infer parameters
init = [1,1,1,1]
init_ps = log.(init)
itera = 2000
results, time, _,_ = @timed Optim.optimize(ps->int_dist(exp.(ps),SSA_PGF,1.0,W,params,re),init_ps,
Optim.Options(show_trace=true,g_tol=1e-11,iterations = itera)).minimizer

# Obtain inferred Parameters
inferred_params = exp.(results)

# Check inferred Parameters
inferred_PGF = BRIDGE_compute_full(inferred_params,params,re)
scatter(SSA_PGF,inferred_PGF,xlabel="SSA",ylabel="inferred");
plot!([minimum(SSA_PGF),maximum(SSA_PGF)],[minimum(SSA_PGF),maximum(SSA_PGF)],lw=2)

