using Plots,Random,Distributions,Flux,DelimitedFiles,FastGaussQuadrature
include("/Users/jiuzong/Documents/GitHub/Cell-FNO/utils.jl")

version = "range4"
ps_true_matrix = readdlm("/Users/jiuzong/Documents/GitHub/Cell-FNO/trailrange/Fig2/data/ps$(version).txt")
σon_list = ps_true_matrix[:,1]
σoff_list = ps_true_matrix[:,2]
ρ_list = ps_true_matrix[:,3]
dm_list = ps_true_matrix[:,4]

# v ?-1
n = 7; a,b = [0,1]

interval_X, weight = gausslegendre(n)
x1 = ((b - a) .* interval_X .+ b .+ a) ./ 2
w1 = weight * (b - a) / 2

interval_X, weight = gausslegendre(n)
x2 = ((b - a) .* interval_X .+ b .+ a) ./ 2
w2 = weight * (b - a) / 2

interval_X, weight = gausslegendre(n)
x3 = ((b - a) .* interval_X .+ b .+ a) ./ 2
w3 = weight * (b - a) / 2

z1 = x1
z2 = x2
z3 = x3
N = length(z1)
batchsize = length(σon_list)

using Flux,DelimitedFiles,Plots
using DataFrames,CSV

hidden_channels = 40 # x1
# hidden_channels = 80 # x2
SSAitera = "SSA1e4"

# load training data
# input: G(z1) = exp(ρ(z1-1))
input = readdlm("/Users/jiuzong/Documents/GitHub/Cell-FNO/trailrange/Fig2/data/matrix_Gz1$(version).csv")
input = vcat(input, reshape(dm_list,1,batchsize))

# output: exp(ρ/λ(z1*z2-1)) $(SSAitera)
train_sol = readdlm("/Users/jiuzong/Documents/GitHub/Cell-FNO/trailrange/Fig2/data/matrix_Gz1z2$(version).csv")
train_sol = reshape(train_sol,(length(z1)*length(z2),batchsize))

# re_select = findall(i -> 0.01 <= ondioff_list[i] <=20
#                     && ps_true_matrix[:,1][i]<4 && ps_true_matrix[:,2][i]<4,1:batchsize)
# ps_true_matrix = ps_true_matrix[re_select,:]

index_list = 1:1:batchsize
# index_list = re_select
input_list = [input[:,i] for i=index_list]
train_sol = train_sol[:,index_list]

# model
model = Chain(Dense(length(z1)+1, hidden_channels,tanh),Dense(hidden_channels, length(z1)*length(z2)),x -> softplus.(x))
params, re = Flux.destructure(model);
ps = Flux.params(params);

# df = DataFrame(params = params)
# CSV.write("/Users/jiuzong/Documents/GitHub/Cell-FNO/trailrange/Fig2/params_trainedx1$(version).txt",df)

function loss_func(p)
    output = re(p).(input_list)
    output = hcat(output...)
    loss = Flux.mse(train_sol,output)

    # output1 = re(p).(input_list[1:400])
    # output1 = hcat(output1...)
    # loss1 = Flux.mse(train_sol[:,1:400],output1)

    # output2 = re(p).(input_list[401:800])
    # output2 = hcat(output2...)
    # loss2 = Flux.mse(train_sol[:,401:800],output2)

    # output3 = re(p).(input_list[801:1200])
    # output3 = hcat(output3...)
    # loss3 = Flux.mse(train_sol[:,801:1200],output3)

    # output4 = re(p).(input_list[1201:1600])
    # output4 = hcat(output4...)
    # loss4 = Flux.mse(train_sol[:,1201:1600],output4)

    # output5 = re(p).(input_list[1601:2000])
    # output5 = hcat(output5...)
    # loss5 = Flux.mse(train_sol[:,1601:2000],output5)

    # loss = (loss1 + loss2 + loss3 + loss4 + loss5)/5

    # output1 = re(p).(input_list[1:400])
    # output1 = hcat(output1...)
    # loss1 = Flux.mse(train_sol[:,1:400],output1)

    # output2 = re(p).(input_list[401:800])
    # output2 = hcat(output2...)
    # loss2 = Flux.mse(train_sol[:,401:800],output2)

    # output3 = re(p).(input_list[801:1200])
    # output3 = hcat(output3...)
    # loss3 = Flux.mse(train_sol[:,801:1200],output3)

    # output4 = re(p).(input_list[1201:1600])
    # output4 = hcat(output4...)
    # loss4 = Flux.mse(train_sol[:,1201:1600],output4)

    # output5 = re(p).(input_list[1601:2000])
    # output5 = hcat(output5...)
    # loss5 = Flux.mse(train_sol[:,1601:2000],output5)

    # output6 = re(p).(input_list[2001:2400])
    # output6 = hcat(output6...)
    # loss6 = Flux.mse(train_sol[:,2001:2400],output6)

    # output7 = re(p).(input_list[2401:2800])
    # output7 = hcat(output7...)
    # loss7 = Flux.mse(train_sol[:,2401:2800],output7)

    # output8 = re(p).(input_list[2801:3207])
    # output8 = hcat(output8...)
    # loss8 = Flux.mse(train_sol[:,2801:3207],output8)

    # output9 = re(p).(input_list[3201:3600])
    # output9 = hcat(output9...)
    # loss9 = Flux.mse(train_sol[:,3201:3600],output9)

    # output10 = re(p).(input_list[3601:3951])
    # output10 = hcat(output10...)
    # loss10 = Flux.mse(train_sol[:,3601:3951],output10)

    # loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8)/8
    return loss
end

using CSV,DataFrames
df = CSV.read("/Users/jiuzong/Documents/GitHub/Cell-FNO/trailrange/Fig2/params_trainedx1$(version).txt",DataFrame)
params = df.params
ps = Flux.params(params);

@time loss_func(params)
mse_min = [loss_func(params)]
@time grads = gradient(()->loss_func(params) , ps)

# training
lr_list = vcat([collect(0.01:-0.001:0.001) for i=1:10]...)
# lr_list = vcat([[0.005;0.004;0.003;0.002;0.001;collect(0.001:-0.0001:0.0001)] for i=1:8]...)
# lr_list1 = vcat([collect(0.01:-0.001:0.001) for i=1:18]...)
# lr_list2 = vcat([[0.005;0.004;0.003;0.002;0.001;collect(0.001:-0.0001:0.0001)] for i=1:12]...)
# lr_list = [lr_list1;lr_list2]
# lr_list = [0.01]

@time for i = 1:length(lr_list)
    lr = lr_list[i]
    opt= ADAM(lr);
    epochs = 2000
    print("interations = ",i,"\n")
    print("learning rate = ",lr,"\n")

    @time for epoch in 1:epochs
        print(epoch,"\n")
        grads = gradient(()->loss_func(params) , ps)
        Flux.update!(opt, ps, grads)

        mse = loss_func(params)
        if mse<mse_min[1]
            df = DataFrame(params = params)
            CSV.write("/Users/jiuzong/Documents/GitHub/Cell-FNO/trailrange/Fig2/params_trainedx1$(version).txt",df)
            mse_min[1] = mse
        end
        print(mse,"\n")
    end

    using CSV,DataFrames
    df = CSV.read("/Users/jiuzong/Documents/GitHub/Cell-FNO/trailrange/Fig2/params_trainedx1$(version).txt",DataFrame)
    params = df.params
    ps = Flux.params(params);
end

using CSV,DataFrames
df = CSV.read("/Users/jiuzong/Documents/GitHub/Cell-FNO/trailrange/Fig2/params_trainedx1$(version).txt",DataFrame);
params = df.params;
ps = Flux.params(params);

@time loss_func(params)
mse_min = [loss_func(params)]



output = re(params).(input_list);
output = hcat(output...)
@time loss_func(params)
train_sol

function plot_distribution(set)
    plot(input[:,set],linewidth = 3,label="SSA",line=:dash)
    plot!(output[:,set],linewidth = 3,label="predict",xlabel = "# of products \n", ylabel = "\n Probability")
    plot!(train_sol[:,set],linewidth = 3,label="SSA",line=:dash)
end

function plot_channel(i)
    p1 = plot_distribution(1+10*(i-1))
    p2 = plot_distribution(2+10*(i-1))
    p3 = plot_distribution(3+10*(i-1))
    p4 = plot_distribution(4+10*(i-1))
    p5 = plot_distribution(5+10*(i-1))
    p6 = plot_distribution(6+10*(i-1))
    p7 = plot_distribution(7+10*(i-1))
    p8 = plot_distribution(8+10*(i-1))
    p9 = plot_distribution(9+10*(i-1))
    p10 = plot_distribution(10+10*(i-1))
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,layouts=(2,5),size=(1500,600))
end
plot_channel(1)

for i = 1:30
    p = plot_channel(i)
    savefig(p,"/Users/jiuzong/Documents/GitHub/Cell-FNO/trailrange/Fig2/train_results/fig_$i.svg")
end

using StatsPlots
rng = Random.seed!(1)
psset = rand(rng,1:1000,20)
true_z1z2value = vec(train_sol[:,psset])
predict_z1z2value = vec(output[:,psset])
qqplot(predict_z1z2value, true_z1z2value, xlabel="Predicted Values",  ylabel="True Values",  
        title="QQ Plot of True vs Predicted Values",markersize = 4,lw=3,size=(500,300))
savefig("/Users/jiuzong/Documents/GitHub/Cell-FNO/trailrange/Fig2/qqplotpre.pdf")