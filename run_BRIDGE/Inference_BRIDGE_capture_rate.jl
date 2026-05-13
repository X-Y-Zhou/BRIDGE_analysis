using Optim, Statistics, Distributions, Plots,StatsBase,DelimitedFiles,Random,Interpolations
using FastGaussQuadrature,Flux,DataFrames,CSV,HypergeometricFunctions,KernelDensity
include("../utils.jl")

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

# Read normalized capture rate β1β2
data = readdlm("dataset/synthetic_data/β1β2.txt")
β1 = data[:, 1]
β2 = data[:, 2]

# Compute joint probability distribution of β1β2 with KDE
Density_joint = kde((β1, β2))
interp_joint = LinearInterpolation((Density_joint.x, Density_joint.y),Density_joint.density,extrapolation_bc = Interpolations.Flat())

# Define the integration interval
min_β1 = minimum(β1)
max_β1 = maximum(β1)

min_β2 = minimum(β2)
max_β2 = maximum(β2)

# Map Gaussian quadrature nodes from [-1, 1] to interval β1, β2.
Xl_β1 = (max_β1 - min_β1) / 2 .* interval_X .+ (max_β1 + min_β1) / 2
Xl_β2 = (max_β2 - min_β2) / 2 .* interval_X .+ (max_β2 + min_β2) / 2

# Calculate the joint probability density p(β1, β2)
pf_joint = [interp_joint(Xl_β1[i], Xl_β2[j]) for i in 1:n, j in 1:n]

# Define hidden channels
hidden_channels = 40

# Initialize BRIGE model
model = Chain(Dense(length(z1)+3, hidden_channels,tanh),Dense(hidden_channels, length(z1)*length(z2)*length(z3)),x -> softplus.(x))
params, re = Flux.destructure(model);
ps = Flux.params(params);

# Define Reduced model PGF with capture rate
function G_tele_delay_cp(σon,σoff,ρ,τ,β,z)
    ρ = ρ*β
    u1 = z-1
    r = 1-ρ*u1+σoff+σon
    θ = sqrt(Complex((ρ*u1-σoff-σon)^2+4*ρ*σon*u1))
    uz = (r+θ-1)/2
    uf = (r-θ-1)/2
    G1 = (uz*exp(-uf*τ)-uf*exp(-uz*τ))/θ + ρ*u1*σon*(exp(-uf*τ)-exp(-uz*τ))/(θ*(σoff+σon))
    return real(G1)
end

# Define the more efficient functions to compute Full model PGF with BRIDGE
function BRIDGE_compute_full_cp(ps,params,re)
    σon, σoff, ρ, dm, λ, dp = ps
    mdl = re(params)
    nβ1 = length(Xl_β1)
    nβ2 = length(Xl_β2)
    L1  = length(z1)

    Gβ1 = G_tele_delay_cp.(σon,σoff,ρ,1.0,reshape(Xl_β1, 1, nβ1),reshape(z1, L1, 1))
    G_tiled = repeat(Gβ1, 1, nβ2)
    dm_row = fill(dm, 1, nβ1 * nβ2)
    dp_row = fill(dp, 1, nβ1 * nβ2)
    λβ2_row = repeat(reshape(λ .* Xl_β2, 1, nβ2),inner = (1, nβ1))
    X = vcat(G_tiled, dm_row, λβ2_row, dp_row)
    Y = mdl(X)
    wmat = pf_joint .* (weights * weights')
    wcol = vec(wmat)
    out = Y * wcol

    scale = ((max_β1 - min_β1) / 2) * ((max_β2 - min_β2) / 2)

    return vec(out .* scale)
end

# Define objective function with capture rate 
function int_dist_cp(ps, SSA_PGF, a, W, params, re)
    dist = BRIDGE_compute_full_cp(ps,params,re).^(1+a) .- BRIDGE_compute_full_cp(ps,params,re).^a .* SSA_PGF .* (1+1/a) .+ SSA_PGF / a
    return sum(W .* dist)
end

# read parameters
using CSV,DataFrames
df = CSV.read("parameters_trained/params_trained3d.txt",DataFrame)
params = df.params
ps = Flux.params(params);

# Read inference counts data
# True value is [σ_on,σ_off,ρ,d,λ,dp] =  [1.011,0.838,3.135,0.208,1.968,1.542]. You can replace it with your own data.
SSA_counts = readdlm("dataset/synthetic_data/counts_example_capture_rate.txt")
N_sample = Int.(SSA_counts[:,1])
M_sample = Int.(SSA_counts[:,2])
P_sample = Int.(SSA_counts[:,3])
Sample_size = length(N_sample)

# Convert counts to PGF
SSA_PGF = convert_counts_to_PGF3d(N_sample,M_sample,P_sample)

# Infer parameters
init = [1,1,1,1,1,1]
init_ps = log.(init)
itera = 1000
results, time, _,_ = @timed Optim.optimize(ps->int_dist_cp(exp.(ps),SSA_PGF,1.0,W,params,re),init_ps,
Optim.Options(show_trace=true,g_tol=1e-20,iterations = itera)).minimizer

# Obtain inferred parameters
inferred_params = exp.(results)

# Check inferred parameters
inferred_PGF = BRIDGE_compute_full_cp(inferred_params,params,re)
Flux.mse(inferred_PGF,SSA_PGF)

scatter(SSA_PGF,inferred_PGF,interval_Xabel="SSA",ylabel="inferred");
plot!([minimum(SSA_PGF),maximum(SSA_PGF)],[minimum(SSA_PGF),maximum(SSA_PGF)],lw=2)

