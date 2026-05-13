# Import packages
using Optim, Statistics, Distributions, Plots,StatsBase,DelimitedFiles,Random
using FastGaussQuadrature,Flux,DataFrames,CSV,HypergeometricFunctions,NLsolve
include("../utils.jl")

# Convert parameters with LMA
function convert_LMA_feedback(ps)
    σ_on,σ_off,ρ,d,λ,dp = ps
    g = σ_on / (σ_off + σ_on)
    mg = λ * ρ * σ_on * (d * dp + d * σ_on + dp * σ_on + σ_off * σ_on + σ_on^2) / (d * dp * (σ_off + σ_on) * (d + σ_off + σ_on) * (dp + σ_off + σ_on))
    σ_off = σ_off*g/mg
    return σ_off
end

# Define Gaussian Quadrature points and corresponding weights
n = 7
a,b = [0,1]
interval_X, weights = gausslegendre(n)
x = ((b - a) .* interval_X .+ b .+ a) ./ 2
w = weights * (b - a) / 2

z1 = x
z2 = x
z3 = x

W = vcat([vec(w*w')*w[i] for i=1:n]...)

# Define hidden channels
hidden_channels = 40

# Initialize BRIGE model
model = Chain(Dense(length(z1)+3, hidden_channels,tanh),Dense(hidden_channels, length(z1)*length(z2)*length(z3)),x -> softplus.(x))
params, re = Flux.destructure(model);
ps = Flux.params(params);

# Read trained parameters
using CSV,DataFrames
df = CSV.read("parameters_trained/params_trained3d.txt",DataFrame)
params = df.params
ps = Flux.params(params);

# Read inference counts data
# True value is [σ_on,σ_off,ρ,d,λ,dp] =  [2.236,0.0252,3.557,0.543,4.452,0.257]. You can replace it with your own data.
SSA_counts = readdlm("dataset/synthetic_data/counts_example_feedback.txt")
U_sample = Int.(SSA_counts[:,1])
S_sample = Int.(SSA_counts[:,2])
P_sample = Int.(SSA_counts[:,3])
Sample_size = length(U_sample)

# Convert counts to PGF
SSA_PGF = convert_counts_to_PGF3d(U_sample,S_sample,P_sample)

# Infer parameters
init = [1,1,1,1,1,1]
init_ps = log.(init)
itera = 1000
results, time, _,_ = @timed Optim.optimize(ps->int_dist(exp.(ps),SSA_PGF,1.0,W, params,re),init_ps,
Optim.Options(show_trace=true,g_tol=1e-20,iterations = itera)).minimizer

# Obtain inferred parameters
inferred_params_equal = exp.(results)

# Convert full model parameters to feedback model parameters
inferred_params = [inferred_params_equal[1];convert_LMA_feedback(inferred_params_equal);inferred_params_equal[3:end]]

# Check inferred parameters
inferred_PGF = BRIDGE_compute_full(inferred_params_equal,params,re)
Flux.mse(inferred_PGF,SSA_PGF)

scatter(SSA_PGF,inferred_PGF,xlabel="SSA",ylabel="inferred");
plot!([minimum(SSA_PGF),maximum(SSA_PGF)],[minimum(SSA_PGF),maximum(SSA_PGF)],lw=2)



