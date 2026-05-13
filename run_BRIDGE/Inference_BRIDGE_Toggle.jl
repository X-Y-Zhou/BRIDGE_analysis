# Import packages
using Optim, Statistics, Distributions, Plots,StatsBase,DelimitedFiles,Random
using FastGaussQuadrature,Flux,DataFrames,CSV,HypergeometricFunctions,NLsolve
include("../utils.jl")

# Convert parameters with LMA for toggle-switch model
function convert_LMA_toggle(ps)
    σ_on1,σ_off1,ρ1,d1,λ1,dp1, 
    σ_on2,σ_off2,ρ2,d2,λ2,dp2 = ps
    g1 = σ_on1 / (σ_off1 + σ_on1)
    g2 = σ_on2 / (σ_off2 + σ_on2)
    m1g2 = λ1 * ρ1 * σ_on1 * σ_on2 / (dp1 * d1 * (σ_off1 + σ_on1) * (σ_off2 + σ_on2))
    m2g1 = λ2 * ρ2 * σ_on1 * σ_on2 / (dp2 * d2 * (σ_off1 + σ_on1) * (σ_off2 + σ_on2))
    σ_off1 = σ_off1*g1/m2g1
    σ_off2 = σ_off2*g2/m1g2
    return [σ_off1,σ_off2]
end

# Convert counts to distribution
function convert_counts_to_PGF3d(U_counts,S_counts,P_counts)
    Sample_size = length(U_counts)
    USP_sample = [[U_counts[i],S_counts[i],P_counts[i]] for i=1:Sample_size]

    U_max = maximum([n for (n, m, p) in USP_sample])
    S_max = maximum([m for (n, m, p) in USP_sample])
    P_max = maximum([p for (n, m, p) in USP_sample])
    
    joint_prob_matrix = zeros(Float64, U_max+1, S_max+1, P_max+1)
    
    for (n, m, p) in USP_sample
        joint_prob_matrix[n+1, m+1, p+1] += 1
    end
    joint_prob_matrix /= length(USP_sample)
    
    SSA_PGF = vec(hist_gf3d(joint_prob_matrix,z1,z2,z3))
    return SSA_PGF
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
# True value is 
# [σ_on1,σ_off1,ρ1,d1,λ1,dp1] = [2.139, 0.108, 1.818, 0.929, 0.566, 1.994]
# [σ_on2,σ_off2,ρ2,d2,λ2,dp2] = [1.523, 0.702, 1.849, 0.941, 3.842, 1.910]
SSA_counts = readdlm("dataset/synthetic_data/counts_example_toggle.txt")
U1_sample = Int.(SSA_counts[:,1])
S1_sample = Int.(SSA_counts[:,2])
P1_sample = Int.(SSA_counts[:,3])

U2_sample = Int.(SSA_counts[:,4])
S2_sample = Int.(SSA_counts[:,5])
P2_sample = Int.(SSA_counts[:,6])

SSA_PGF1 = convert_counts_to_PGF3d(U1_sample,S1_sample,P1_sample)
SSA_PGF2 = convert_counts_to_PGF3d(U2_sample,S2_sample,P2_sample)

# Infer parameters
init = [1,1,1,1,1,1]
init_ps = log.(init)
itera = 1000

results1, time1, _,_ = @timed Optim.optimize(ps->int_dist(exp.(ps),SSA_PGF1,1.0,W,params,re),init_ps,
Optim.Options(show_trace=true,g_tol=1e-20,iterations = itera)).minimizer

results2, time2, _,_ = @timed Optim.optimize(ps->int_dist(exp.(ps),SSA_PGF2,1.0,W,params,re),init_ps,
Optim.Options(show_trace=true,g_tol=1e-20,iterations = itera)).minimizer

# Obtain inferred parameters
inferred_params_equal1 = exp.(results1)
inferred_params_equal2 = exp.(results2)

# Convert full model parameters to toggle-switch model parameters
soff_all = convert_LMA_toggle([inferred_params_equal1;inferred_params_equal2])
inferred_params = [inferred_params_equal1[1];soff_all[1];inferred_params_equal1[3:end];
                   inferred_params_equal2[1];soff_all[2];inferred_params_equal2[3:end];]

# Check inferred parameters
inferred_PGF1 = BRIDGE_compute_full(inferred_params_equal1,params,re)
Flux.mse(inferred_PGF1,SSA_PGF1)

inferred_PGF2 = BRIDGE_compute_full(inferred_params_equal2,params,re)
Flux.mse(inferred_PGF2,SSA_PGF2)

p1 = scatter(SSA_PGF1,inferred_PGF1,xlabel="SSA",ylabel="inferred");
plot!([minimum(SSA_PGF1),maximum(SSA_PGF1)],[minimum(SSA_PGF1),maximum(SSA_PGF1)],lw=2);

p2 = scatter(SSA_PGF2,inferred_PGF2,xlabel="SSA",ylabel="inferred");
plot!([minimum(SSA_PGF2),maximum(SSA_PGF2)],[minimum(SSA_PGF2),maximum(SSA_PGF2)],lw=2);

plot(p1,p2,size=(1200,600))
