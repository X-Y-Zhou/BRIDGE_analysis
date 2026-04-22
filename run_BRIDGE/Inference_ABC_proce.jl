using Distributed,Pkg
addprocs(10)
nprocs()
workers()

@everywhere using Statistics, Distributions, Plots,StatsBase,DelimitedFiles,Random
@everywhere using Flux,DataFrames,CSV
@everywhere using Catalyst.EnsembleAnalysis,Statistics
@everywhere using DelaySSAToolkit,Catalyst,ApproxBayes
@everywhere include("/Users/x-y-zhou/Documents/GitHub/Cell-FNO/utilsv2.jl")

@everywhere function synthetic_SSA(rng,joint,Sample_size,N1)
    joint = vec(joint)
    joint = abs.(joint)./sum(abs.(joint))
    joint_probability = Categorical(joint)
    index = rand(rng,joint_probability,Sample_size).-1

    # row
    count_N = rem.(index,N1)

    # column
    count_M = div.(index,N1)

    return [count_N,count_M]
end

@everywhere rn = @reaction_network begin
    σon,  Goff --> Gon
    σoff, Gon  --> Goff
    ρ, Gon --> Gon + N
    dm, M --> 0
# end
end σon σoff ρ dm

@everywhere jumpsys = convert(JumpSystem, rn; combinatoric_ratelaws=false)

@everywhere u0 = [1, 0, 0, 0]
@everywhere de_chan0 = [[]]
@everywhere tf = 15.0
@everywhere tspan = (0, tf)

@everywhere version = "forinferv2"
@everywhere SSA_iteration = "1e4"

@everywhere ps_true_matrix = readdlm("/Users/x-y-zhou/Documents/GitHub/Cell-FNO/trailrange/Fig2/data/ps$(version).txt")
@everywhere sample_size = size(ps_true_matrix,1)

@everywhere meanM_list = ps_true_matrix[:,1].*ps_true_matrix[:,3]./(ps_true_matrix[:,2].+ps_true_matrix[:,1])
@everywhere meanN_list = ps_true_matrix[:,1].*ps_true_matrix[:,3].*ps_true_matrix[:,4]./(ps_true_matrix[:,2].+ps_true_matrix[:,1])
@everywhere ondioff_list = ps_true_matrix[:,1]./ps_true_matrix[:,2]
@everywhere re_select = findall(i -> 0.01 <= ondioff_list[i] <=20
                    && ps_true_matrix[:,1][i]<4 && ps_true_matrix[:,2][i]<4,1:sample_size)

@everywhere ps_true_matrix = ps_true_matrix[re_select,:]

# read SSA data
@everywhere SSA_data_all = restore_nested_array("/Users/x-y-zhou/Documents/GitHub/Cell-FNO/trailrange/Fig2/data/SSA$(version)_$(SSA_iteration).csv")[re_select]
@everywhere FSP_data_all = restore_nested_array("/Users/x-y-zhou/Documents/GitHub/Cell-FNO/trailrange/Fig2/data/FSP$(version).csv")[re_select]

# set = 4
# maximum(ps_true_matrix[:,set])
# minimum(ps_true_matrix[:,set])

@everywhere function bayesdist(params, constant, SSA_counts)
    p = params
    τ = 1

    dprob = DiscreteProblem(u0, tspan, p)
    delay_trigger_affect! = function (integrator, rng)
        append!(integrator.de_chan[1], τ)
    end
    delay_trigger = Dict(3 => delay_trigger_affect!)
    delay_complete = Dict(1 => [3 => -1, 4 => 1])
    delay_interrupt = Dict()
    delayjumpset = DelayJumpSet(delay_trigger, delay_complete, delay_interrupt)

    djprob = DelayJumpProblem(
        jumpsys, dprob, DelayRejection(), delayjumpset, de_chan0; save_positions=(false, false)
    )

    interation = Int(1e4)
    ensprob = EnsembleProblem(djprob)
    ens = solve(ensprob, SSAStepper(), EnsembleThreads(); trajectories=interation,saveat=1)
    N_sample = componentwise_vectors_timepoint(ens, tf)[3]
    M_sample = componentwise_vectors_timepoint(ens, tf)[4]
    simdata = [N_sample;M_sample]
    ApproxBayes.ksdist(simdata, SSA_counts), 1

    # return convert_histo(N_sample)[2],convert_histo(M_sample)[2]
end

# set = 124
# params = ps_true_matrix[set,:]

# # SSA with FSP
# Sample_size = Int(1e4)
# rng = Random.seed!(1)
# joint_probability = FSP_data_all[set]
# N1,N2 = size(joint_probability)
# count_N,count_M = synthetic_SSA(rng,joint_probability,Sample_size,N1)
# prob_N1 = convert_histo(count_N)[2]
# prob_M1 = convert_histo(count_M)[2]
# SSA_counts = [count_N;count_M]

# # SSA with simulation
# prob_N,prob_M = bayesdist(params, 1, SSA_counts)

# # FSP
# joint_FSP_true = FSP_data_all[set]
# P_N_true = vec(sum(joint_FSP_true,dims=2))
# P_M_true = vec(sum(joint_FSP_true,dims=1))

# plot(0:length(prob_N)-1, prob_N,lw=3)
# plot!(0:N1-1, P_N_true,lw=3,line=:dash)
# plot!(0:length(prob_N1)-1, prob_N1,lw=3,line=:dash)

# plot(0:length(prob_M)-1, prob_M,lw=3)
# plot!(0:N2-1, P_M_true,lw=3,line=:dash)
# plot!(0:length(prob_M1)-1, prob_M1,lw=3,line=:dash)

@everywhere function RunABC(set)
    print(set,"\n")
    Sample_size = Int(1e4)
    rng = Random.seed!(1)
    joint_probability = FSP_data_all[set]
    N1,N2 = size(joint_probability)
    count_N,count_M = synthetic_SSA(rng,joint_probability,Sample_size,N1)
    SSA_counts = [count_N;count_M]
    ϵ = 0.2

    setup = ABCRejection(bayesdist, #simulation function
        4, # number of parameters
        ϵ, #target 
        ApproxBayes.Prior([Uniform(0, 4.0), Uniform(0, 4.0),Uniform(0.0, 50.0),Uniform(0, 2),]); # Prior for each of the parameters
    )

    results, time, _,_ = @timed runabc(setup, SSA_counts,progress=false)
    return [vec((median(results.parameters,dims=1)));time]
end

batchsize = length(SSA_data_all)
# batchsize = 1
@time params_infered_list = pmap(set->RunABC(set),201:287)

writedlm("/Users/x-y-zhou/Documents/GitHub/Cell-FNO/trailrange/Fig2/inference_ABC/" *
                "infer_params$(version)201-287.txt",params_infered_list)




