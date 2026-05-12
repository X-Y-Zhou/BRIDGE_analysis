# Inference with MOM

using Optim, CSV, DataFrames,LinearAlgebra, Distributions, SparseArrays
using Statistics,DelimitedFiles,StatsBase,Distributed
include("utils.jl")

function observed_moments(N_sample,M_sample)
    mean_N = mean(N_sample)
    mean_M = mean(M_sample)
    var_N = var(N_sample)
    var_M = var(M_sample)
    return [mean_N,mean_M,var_N,var_M]
end

# FSP 模拟函数
function simulate_moments(ps)
    σon,σoff,ρ,d = ps
    τ = 1
    
    # mean and variance
    mean_N=ρ*σon*τ/(σoff+σon)
    mean_M=ρ*σon/((σoff+σon)*d)

    var_N=ρ*σon*τ/(σoff+σon)+2*ρ^2*σon*σoff*(-1+exp(-(σoff+σon)*τ)+(σoff+σon)*τ)/((σoff+σon)^4)
    var_M=ρ*σon/(d*(σoff+σon))+ρ^2*σon*σoff/(d*(σoff+σon)^2*(d+σoff+σon))

    simulated_data=[mean_N,mean_M,var_N,var_M]
    return simulated_data
end

function objective_function(ps,SSA_counts)
    simulated_data = simulate_moments(ps)

    N_sample = Int.(SSA_counts[:,1])
    M_sample = Int.(SSA_counts[:,2])
    observed_data = observed_moments(N_sample,M_sample)

    mse = sum(((observed_data .- simulated_data)./observed_data).^2) / length(observed_data)
    return mse
end

SSA_counts = readdlm("dataset/synthetic_data/counts_example2d.txt")
init = [1,1,1,1]
init_ps = log.(init)
itera = 1000
results, time, _,_ = @timed Optim.optimize(ps -> objective_function(exp.(ps), SSA_counts),
                                            init_ps,Optim.Options(show_trace=true,g_tol=1e-11,iterations = itera)).minimizer

infer_params = exp.(results)






