using Distributed,Pkg
addprocs(10)
nprocs()
workers()

@everywhere using DelimitedFiles,Optim,Flux
@everywhere include("/Users/x-y-zhou/Documents/GitHub/Cell-FNO/trailrange/Fig2/data/FSP_function.jl")
@everywhere include("/Users/x-y-zhou/Documents/GitHub/Cell-FNO/utilsv2.jl")

@everywhere function resize_matrix(mat::AbstractMatrix, N1::Int, N2::Int)
    M1, M2 = size(mat)
    new_mat = zeros(eltype(mat), N1, N2)
    r1 = min(N1, M1)
    r2 = min(N2, M2)
    new_mat[1:r1, 1:r2] .= mat[1:r1, 1:r2]
    return new_mat
end

@everywhere function int_dist(ps, hist_data)
    σon,σoff,ρ,d = ps
    τ = 1
    pt = [σon,σoff,ρ,τ,d]
    # N_max,M_max = size(hist_data)
    N_max,M_max = [60,60]
    hist_data = resize_matrix(hist_data,N_max,M_max)
    return Flux.mse(hist_data,delaysolG1(pt,N_max, M_max))
end

# trailrange/Fig2/inference_FSP/infer_paramsforinferv2.txt:
# N_max,M_max = [60,60]
# hist_data = resize_matrix(hist_data,N_max,M_max)

# trailrange/Fig2/inference_FSP/infer_paramsforinfer.txt:
# N_max,M_max = size(hist_data)

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

# read SSA data
@everywhere SSA_data_all = restore_nested_array("/Users/x-y-zhou/Documents/GitHub/Cell-FNO/trailrange/Fig2/data/SSA$(version)_$(SSA_iteration).csv")[re_select]
# maximum([size(SSA_data_all[i],1) for i=1:length(SSA_data_all)])
# maximum([size(SSA_data_all[i],2) for i=1:length(SSA_data_all)])

@everywhere function estimate(set)
    print(set,"\n")
    hist_data = SSA_data_all[set]

    init = [1,1,1,1]
    init_ps = log.(init)
    itera = 1000
    results, time, _,_ = @timed Optim.optimize(ps->int_dist(exp.(ps),hist_data),init_ps,
    Optim.Options(show_trace=false,g_tol=1e-11,iterations = itera)).minimizer

    infer_params = exp.(results)
    return [infer_params;time]
end

version
SSA_iteration
# batchsize = length(SSA_data_all)
# batchsize = 200
@time params_infered_list = pmap(set->estimate(set),301:400)


writedlm("/Users/x-y-zhou/Documents/GitHub/Cell-FNO/trailrange/Fig2/inference_FSP/" *
                "infer_params$(version)401-300.txt",params_infered_list)


ps_forinfer1 = readdlm("trailrange/Fig2/inference_FSP/infer_paramsforinfer1-200.txt")
ps_forinfer2 = readdlm("trailrange/Fig2/inference_FSP/infer_paramsforinfer201-300.txt")
ps_forinfer3 = readdlm("trailrange/Fig2/inference_FSP/infer_paramsforinfer401-300.txt")

ps_forinfer = vcat(ps_forinfer1,ps_forinfer2,ps_forinfer3)
writedlm("trailrange/Fig2/inference_FSP/infer_paramsforinferv2.txt",ps_forinfer)

