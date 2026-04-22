using Distributed,Pkg
addprocs(10)
nprocs()
workers()

@everywhere include("/Users/x-y-zhou/Documents/GitHub/Cell-FNO/utilsv2.jl")
@everywhere function Gz1z2(σon,σoff,ρ,τ,dm,z1,z2)
    τ=1
    u1=z1-1
    u2=z2-1
    x1=ρ*u1/dm
    x2=ρ*u2/dm
    α=σon/(σoff+σon)
    r=1+(σoff+σon)/dm-x1
    θ=sqrt(Complex(((σoff+σon)/dm-x1)^2+4*σon*x1/dm))
    v1=dm*(r+θ-1)/2
    v2=dm*(r-θ-1)/2
    G1=(v1*exp(-v2*τ)-v2*exp(-v1*τ))/(dm*θ)*pFq((σon/dm, ), ((σon+σoff)/dm, ), x2)+ρ*u1*(exp(-v2*τ)-exp(-v1*τ))/(dm*θ)*σon/(σoff+σon)*pFq((1+σon/dm, ), (1+(σoff+σon)/dm, ), x2)
    return real(G1)
end

@everywhere using Optim, Statistics, Distributions, Plots,StatsBase,DelimitedFiles,Random
@everywhere using FastGaussQuadrature,Flux,DataFrames,CSV,HypergeometricFunctions

@everywhere n = 7
@everywhere a = 0
@everywhere b = 1
@everywhere interval_X, weights = gausslegendre(n)
@everywhere x1 = ((b - a) .* interval_X .+ b .+ a) ./ 2
@everywhere w1 = weights * (b - a) / 2

@everywhere a = 0
@everywhere b = 1
@everywhere interval_X, weights = gausslegendre(n)
@everywhere x2 = ((b - a) .* interval_X .+ b .+ a) ./ 2
@everywhere w2 = weights * (b - a) / 2

@everywhere X = [(x1[i],x2[j]) for i = 1 :length(x1) for j = 1:length(x2)]
@everywhere W = vec(w1*w2')

@everywhere z1 = x1
@everywhere z2 = x2

@everywhere Z1 = repeat(z1, 1, length(z2))   # 行重复
@everywhere Z2 = repeat(z2', length(z1), 1)  # 列重复（注意转置）
# @time Gvals = Gz1z2.(σon_list[i],σoff_list[i],ρ_list[i],1,dm_list[i], Z1, Z2)

@everywhere function get_exact_gf(ps)
    σon,σoff,ρ,dm = ps
    z_mat = Gz1z2.(σon,σoff,ρ,1,dm, Z1, Z2)
    return vec(z_mat)
end

@everywhere function int_dist(ps, SSA_PGF, a, W)
    dist = get_exact_gf(ps).^(1+a) .- get_exact_gf(ps).^a .* SSA_PGF .* (1+1/a) .+ SSA_PGF / a
    return sum(W .* dist)
end


@everywhere version = "forinfer"
@everywhere SSA_iteration = "1e4"

@everywhere ps_true_matrix = readdlm("/Users/x-y-zhou/Documents/GitHub/Cell-FNO/trailrange/Fig2/data/ps$(version).txt")
@everywhere sample_size = size(ps_true_matrix,1)

@everywhere meanM_list = ps_true_matrix[:,1].*ps_true_matrix[:,3]./(ps_true_matrix[:,2].+ps_true_matrix[:,1])
@everywhere meanN_list = ps_true_matrix[:,1].*ps_true_matrix[:,3].*ps_true_matrix[:,4]./(ps_true_matrix[:,2].+ps_true_matrix[:,1])
@everywhere ondioff_list = ps_true_matrix[:,1]./ps_true_matrix[:,2]
@everywhere re_select = findall(i -> 0.01 <= ondioff_list[i] <=20
                        && ps_true_matrix[:,1][i]<4 && ps_true_matrix[:,2][i]<4,1:sample_size)

# @everywhere re_select = 1:1:sample_size
@everywhere ps_true_matrix = ps_true_matrix[re_select,:]
ps_true_matrix

# read SSA data
@everywhere SSA_data_all = restore_nested_array("/Users/x-y-zhou/Documents/GitHub/Cell-FNO/trailrange/Fig2/data/SSA$(version)_$(SSA_iteration).csv")[re_select]
@everywhere SSA_data_PGF = [vec(hist_gf2d(SSA_data_all[i],z1,z2)) for i=1:length(SSA_data_all)]

@everywhere function estimate(set)
    print(set,"\n")
    SSA_PGF = SSA_data_PGF[set]

    init = [1,1,1,1]
    init_ps = log.(init)
    itera = 1000
    results, time, _,_ = @timed Optim.optimize(ps->int_dist(exp.(ps),SSA_PGF,1.0,W),init_ps,
    Optim.Options(show_trace=false,g_tol=1e-11,iterations = itera)).minimizer

    infer_params = exp.(results)
    return [infer_params;time]
end

version
SSA_iteration
length(SSA_data_all)
@time params_infered_list = pmap(set->estimate(set),1:length(SSA_data_all))

writedlm("/Users/x-y-zhou/Documents/GitHub/Cell-FNO/trailrange/Fig2/inference_exact/" *
                "infer_params$(version).txt",params_infered_list)




