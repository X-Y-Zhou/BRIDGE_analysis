# Import packages
using Optim, Statistics, Distributions, Plots,StatsBase,DelimitedFiles,Random
using FastGaussQuadrature,Flux,DataFrames,CSV,HypergeometricFunctions,NLsolve
include("../utils.jl")

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

function convert_counts_to_PGF3d(U_counts,S_counts,P_counts)
    Sample_size = length(U_counts)
    NMP_sample = [[U_counts[i],S_counts[i],P_counts[i]] for i=1:Sample_size]

    N_max = maximum([n for (n, m, p) in NMP_sample])
    M_max = maximum([m for (n, m, p) in NMP_sample])
    P_max = maximum([p for (n, m, p) in NMP_sample])
    
    joint_prob_matrix = zeros(Float64, N_max+1, M_max+1, P_max+1)
    
    for (n, m, p) in NMP_sample
        joint_prob_matrix[n+1, m+1, p+1] += 1
    end
    joint_prob_matrix /= length(NMP_sample)
    
    SSA_PGF = vec(hist_gf3d(joint_prob_matrix,z1,z2,z3))
    return SSA_PGF
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
# True value is 
# [σ_on1,σ_off1,ρ1,d1,λ1,dp1] = [2.139, 0.108, 1.818, 0.929, 0.566, 1.994]
# [σ_on2,σ_off2,ρ2,d2,λ2,dp2] = [1.523, 0.702, 1.849, 0.941, 3.842, 1.910]
SSA_counts = readdlm("dataset/synthetic_data/counts_example_toggle.txt")
N1_sample = Int.(SSA_counts[:,1])
M1_sample = Int.(SSA_counts[:,2])
P1_sample = Int.(SSA_counts[:,3])

N2_sample = Int.(SSA_counts[:,4])
M2_sample = Int.(SSA_counts[:,5])
P2_sample = Int.(SSA_counts[:,6])

SSA_PGF1 = convert_counts_to_PGF3d(N1_sample,M1_sample,P1_sample)
SSA_PGF2 = convert_counts_to_PGF3d(N2_sample,M2_sample,P2_sample)

# Infer parameters
init = [1,1,1,1,1,1]
init_ps = log.(init)
itera = 1000

results1, time1, _,_ = @timed Optim.optimize(ps->int_dist(exp.(ps),SSA_PGF1,1.0,W),init_ps,
Optim.Options(show_trace=true,g_tol=1e-20,iterations = itera)).minimizer

results2, time2, _,_ = @timed Optim.optimize(ps->int_dist(exp.(ps),SSA_PGF2,1.0,W),init_ps,
Optim.Options(show_trace=true,g_tol=1e-20,iterations = itera)).minimizer

# Obtain inferred parameters
inferred_params_equal1 = exp.(results1)
inferred_params_equal2 = exp.(results2)

# Convert full model parameters to toggle-switch model parameters
soff_all = convert_LMA_toggle([inferred_params_equal1;inferred_params_equal2])
inferred_params = [inferred_params_equal1[1];soff_all[1];inferred_params_equal1[3:end];
                   inferred_params_equal2[1];soff_all[2];inferred_params_equal2[3:end];]

# Check inferred parameters
inferred_PGF1 = get_MLP_gf(inferred_params_equal1)
Flux.mse(inferred_PGF1,SSA_PGF1)

inferred_PGF2 = get_MLP_gf(inferred_params_equal2)
Flux.mse(inferred_PGF2,SSA_PGF2)


p1 = scatter(SSA_PGF1,inferred_PGF1,xlabel="SSA",ylabel="inferred");
plot!([minimum(SSA_PGF1),maximum(SSA_PGF1)],[minimum(SSA_PGF1),maximum(SSA_PGF1)],lw=2);

p2 = scatter(SSA_PGF2,inferred_PGF2,xlabel="SSA",ylabel="inferred");
plot!([minimum(SSA_PGF2),maximum(SSA_PGF2)],[minimum(SSA_PGF2),maximum(SSA_PGF2)],lw=2);

plot(p1,p2,size=(1200,600))



set = 63
re_list = []
ps_inferred_list = []
set_list = Int.(vec(readdlm("/Users/x-y-zhou/Documents/GitHub/Cell-FNO/trailrange/Fig4_LMA/inference/inference_results/plot_omni/re_good.txt")))
for set in set_list
    print(set,"\n")
    SSA_counts = readdlm("dataset/synthetic_data/toggle/counts_example_toggle$(set).txt")
    N1_sample = Int.(SSA_counts[:,1])
    M1_sample = Int.(SSA_counts[:,2])
    P1_sample = Int.(SSA_counts[:,3])

    N2_sample = Int.(SSA_counts[:,4])
    M2_sample = Int.(SSA_counts[:,5])
    P2_sample = Int.(SSA_counts[:,6])

    SSA_PGF1 = convert_counts_to_PGF3d(N1_sample,M1_sample,P1_sample)
    SSA_PGF2 = convert_counts_to_PGF3d(N2_sample,M2_sample,P2_sample)

    # Infer parameters
    init = [1,1,1,1,1,1]
    init_ps = log.(init)
    itera = 1000

    results1, time1, _,_ = @timed Optim.optimize(ps->int_dist(exp.(ps),SSA_PGF1,1.0,W),init_ps,
    Optim.Options(show_trace=false,g_tol=1e-20,iterations = itera)).minimizer

    results2, time2, _,_ = @timed Optim.optimize(ps->int_dist(exp.(ps),SSA_PGF2,1.0,W),init_ps,
    Optim.Options(show_trace=false,g_tol=1e-20,iterations = itera)).minimizer

    ps_true_matrix_toggle = readdlm("/Users/x-y-zhou/Documents/GitHub/Cell-FNO/trailrange/Fig4_LMA/inference/inference_results/plot_omni/params_true_toggle.txt")
    ps_true = 10 .^ps_true_matrix_toggle[set,:]

    # Obtain inferred parameters
    inferred_params_equal1 = exp.(results1)
    inferred_params_equal2 = exp.(results2)

    soff_all = convert_LMA_toggle([inferred_params_equal1;inferred_params_equal2])
    inferred_params = [inferred_params_equal1[1];soff_all[1];inferred_params_equal1[3:end];
                    inferred_params_equal2[1];soff_all[2];inferred_params_equal2[3:end];]

    push!(re_list,mean(abs.(inferred_params.-ps_true)./ps_true))
    push!(ps_inferred_list,inferred_params)
end

re_list

re_list[16]
ps_inferred_list[16]
set_list[16]


