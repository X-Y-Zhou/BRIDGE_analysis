# Import packages
using Optim, Statistics, Distributions, Plots,StatsBase,DelimitedFiles,Random
using FastGaussQuadrature,Flux,DataFrames,CSV,HypergeometricFunctions
include("utils.jl")

# Define Reduced model PGF
function G_tele_delay(σon,σoff,ρ,τ,z)
    u1 = z-1
    r = 1-ρ*u1+σoff+σon
    θ = sqrt(Complex((ρ*u1-σoff-σon)^2+4*ρ*σon*u1))
    uz = (r+θ-1)/2
    uf = (r-θ-1)/2
    G1 = (uz*exp(-uf*τ)-uf*exp(-uz*τ))/θ + ρ*u1*σon*(exp(-uf*τ)-exp(-uz*τ))/(θ*(σoff+σon))
    return real(G1)
end

# Define the function to compute Full model PGF with BRIDGE
function forward(x,λ_list,p)
    x = vcat(x, λ_list)
    output = re(p)(x)
    return output
end

function MLP_gf(input,λ_list,p)
    params = p
    output = forward(input,λ_list,params)
    return output
end

function get_MLP_gf(ps)
    σon,σoff,ρ,dm,λ,dp = ps
    input = G_tele_delay.(σon,σoff,ρ,1.,z1)
    output = MLP_gf(input,[dm,λ,dp],params)
    return output
end

# Define objective function
function int_dist(ps, SSA_PGF, a, W)
    dist = get_MLP_gf(ps).^(1+a) .- get_MLP_gf(ps).^a .* SSA_PGF .* (1+1/a) .+ SSA_PGF / a
    return sum(W .* dist)
end

# Define Gaussian Quadrature points and corresponding weights
n = 7
a = 0
b = 1
interval_X, weights = gausslegendre(n)
x1 = ((b - a) .* interval_X .+ b .+ a) ./ 2
w1 = weights * (b - a) / 2

a = 0
b = 1
interval_X, weights = gausslegendre(n)
x2 = ((b - a) .* interval_X .+ b .+ a) ./ 2
w2 = weights * (b - a) / 2

a = 0
b = 1
interval_X, weights = gausslegendre(n)
x3 = ((b - a) .* interval_X .+ b .+ a) ./ 2
w3 = weights * (b - a) / 2

W = vcat([vec(w1*w2')*w3[i] for i=1:n]...)

z1 = x1
z2 = x2
z3 = x3

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
# True value is [σ_on,σ_off,ρ,d,λ,dp] =  [1.773,0.410,1.528,0.973,4.529,1.241]. You can replace it with your own data.
SSA_counts = readdlm("dataset/synthetic_data/counts_example3d.txt")
N_sample = Int.(SSA_counts[:,1])
M_sample = Int.(SSA_counts[:,2])
P_sample = Int.(SSA_counts[:,3])
Sample_size = length(N_sample)

# Convert counts data to joint distribution
NMP_sample = [[N_sample[i],M_sample[i],P_sample[i]] for i=1:Sample_size]

N_max = maximum([n for (n, m, p) in NMP_sample])
M_max = maximum([m for (n, m, p) in NMP_sample])
P_max = maximum([p for (n, m, p) in NMP_sample])

joint_prob_matrix = zeros(Float64, N_max+1, M_max+1, P_max+1)

for (n, m, p) in NMP_sample
    joint_prob_matrix[n+1, m+1, p+1] += 1
end
joint_prob_matrix /= length(NMP_sample)

# Convert joint distribution to PGF
SSA_PGF = vec(hist_gf3d(joint_prob_matrix,z1,z2,z3))

# Infer parameters
init = [1,1,1,1,1,1]
init_ps = log.(init)
itera = 1000
results, time, _,_ = @timed Optim.optimize(ps->int_dist(exp.(ps),SSA_PGF,1.0,W),init_ps,
Optim.Options(show_trace=true,g_tol=1e-20,iterations = itera)).minimizer

# Obtain inferred Parameters
inferred_params = exp.(results)

# Check inferred Parameters
inferred_PGF = get_MLP_gf(inferred_params)
Flux.mse(inferred_PGF,SSA_PGF)

scatter(SSA_PGF,inferred_PGF);
plot!([minimum(SSA_PGF),maximum(SSA_PGF)],[minimum(SSA_PGF),maximum(SSA_PGF)],lw=2)
