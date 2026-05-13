using Optim, Statistics, Distributions, Plots,StatsBase,DelimitedFiles,Random,Interpolations
using FastGaussQuadrature,Flux,DataFrames,CSV,HypergeometricFunctions,KernelDensity
include("../utils.jl")

function G_tele_delay(σon,σoff,ρ,τ,β,z)
    ρ = ρ*β
    u1 = z-1
    r = 1-ρ*u1+σoff+σon
    θ = sqrt(Complex((ρ*u1-σoff-σon)^2+4*ρ*σon*u1))
    uz = (r+θ-1)/2
    uf = (r-θ-1)/2
    G1 = (uz*exp(-uf*τ)-uf*exp(-uz*τ))/θ + ρ*u1*σon*(exp(-uf*τ)-exp(-uz*τ))/(θ*(σoff+σon))
    return real(G1)
end

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
Ny = 7
xl, wl = gausslegendre(Ny)

data = readdlm("dataset/synthetic_data/βλforinfercv_1e4.txt")
β1 = data[:, 1]
β2 = data[:, 2]
Density_joint = kde((β1, β2))

interp_joint = LinearInterpolation(
    (Density_joint.x, Density_joint.y),
    Density_joint.density,
    extrapolation_bc = Interpolations.Flat()
)

# 3. 积分区间
min_β1 = minimum(β1)
max_β1 = maximum(β1)

min_β2 = minimum(β2)
max_β2 = maximum(β2)

# 4. Gauss 节点从 [-1, 1] 映射到真实 β1, β2 区间
Xl_β1 = (max_β1 - min_β1) / 2 .* xl .+ (max_β1 + min_β1) / 2
Xl_β2 = (max_β2 - min_β2) / 2 .* xl .+ (max_β2 + min_β2) / 2

# 5. 在二维 Gauss 节点上计算联合概率密度 p(β1, β2)
pf_joint = [
    interp_joint(Xl_β1[i], Xl_β2[j])
    for i in 1:Ny, j in 1:Ny
]

height = length(z2)+1
in_channels = 1
hidden_channels = 40
N = height
channels = 1

model = Chain(Dense(length(z1)+3, hidden_channels,tanh),Dense(hidden_channels, length(z1)*length(z2)*length(z3)),x -> softplus.(x))
params, re = Flux.destructure(model);
ps = Flux.params(params);

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
    σon, σoff, ρ, dm, λ, dp = ps
    mdl = re(params)
    nβ1 = length(Xl_β1)
    nβ2 = length(Xl_β2)
    L1  = length(z1)

    Gβ1 = G_tele_delay.(σon,σoff,ρ,1.0,reshape(Xl_β1, 1, nβ1),reshape(z1, L1, 1))
    G_tiled = repeat(Gβ1, 1, nβ2)
    dm_row = fill(dm, 1, nβ1 * nβ2)
    dp_row = fill(dp, 1, nβ1 * nβ2)
    λβ2_row = repeat(reshape(λ .* Xl_β2, 1, nβ2),inner = (1, nβ1))
    X = vcat(G_tiled, dm_row, λβ2_row, dp_row)
    Y = mdl(X)
    wmat = pf_joint .* (wl * wl')
    wcol = vec(wmat)
    out = Y * wcol

    scale = ((max_β1 - min_β1) / 2) * ((max_β2 - min_β2) / 2)

    return vec(out .* scale)
end

function int_dist(ps, SSA_PGF, a, W)
    dist = get_MLP_gf(ps).^(1+a) .- get_MLP_gf(ps).^a .* SSA_PGF .* (1+1/a) .+ SSA_PGF / a
    return sum(W .* dist)
end

# read parameters
using CSV,DataFrames
df = CSV.read("parameters_trained/params_trained3d.txt",DataFrame)
params = df.params
ps = Flux.params(params);

# Read inference counts data
# True value is [σ_on,σ_off,ρ,d,λ,dp] =  [1.773,0.410,1.528,0.973,4.529,1.241]. You can replace it with your own data.
set = 1832
SSA_counts = readdlm("dataset/synthetic_data/capture_rate/counts_example_capture_rate$(set).txt")
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

# Obtain inferred parameters
inferred_params = exp.(results)

ps_true_matrix = readdlm("/Users/x-y-zhou/Documents/GitHub/Cell-FNO/trailrange/Fig5/inference_CV/psforinfercv.txt")
[ps_true_matrix[set,:] inferred_params]
mean(abs.(inferred_params.-ps)./ps)

# Check inferred parameters
inferred_PGF = get_MLP_gf(inferred_params)
Flux.mse(inferred_PGF,SSA_PGF)

scatter(SSA_PGF,inferred_PGF,xlabel="SSA",ylabel="inferred");
plot!([minimum(SSA_PGF),maximum(SSA_PGF)],[minimum(SSA_PGF),maximum(SSA_PGF)],lw=2)

re_list = []
ps_inferred_list = []
set_list = Int.(vec(readdlm("/Users/x-y-zhou/Documents/GitHub/Cell-FNO/trailrange/Fig5/inference_CV/inference_results/re_good3.txt")))

for set in set_list
    print(set,"\n")
    SSA_counts = readdlm("dataset/synthetic_data/capture_rate/counts_example_capture_rate$(set).txt")
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
    Optim.Options(show_trace=false,g_tol=1e-20,iterations = itera)).minimizer

    # Obtain inferred parameters
    inferred_params = exp.(results)

    ps_true_matrix = readdlm("/Users/x-y-zhou/Documents/GitHub/Cell-FNO/trailrange/Fig5/inference_CV/psforinfercv.txt")
    ps_true = ps_true_matrix[set,:]
    push!(re_list,mean(abs.(inferred_params.-ps_true)./ps_true))
    push!(ps_inferred_list,inferred_params)
end

re_list[29]
ps_inferred_list
set_list[29]
