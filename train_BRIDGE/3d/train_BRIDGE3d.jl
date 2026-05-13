# Load packages
using Plots,Random,Distributions,Flux,DelimitedFiles,FastGaussQuadrature
using Flux,DelimitedFiles,Plots
using DataFrames,CSV
include("../../utils.jl")

# Read kinectic parameters for training
ps_true_matrix = readdlm("train_BRIDGE/3d/ps_for_train.txt")
σon_list = ps_true_matrix[:,1]
σoff_list = ps_true_matrix[:,2]
ρ_list = ps_true_matrix[:,3]
dm_list = ps_true_matrix[:,4]
λ_list = ps_true_matrix[:,5]
dp_list = ps_true_matrix[:,6]
batchsize = length(σon_list)

# Define number of Gaussian Quadrature points 
n = length(z1)

# Define Gaussian Quadrature points and corresponding weights
a,b = [0,1]
interval_X, weight = gausslegendre(n)
x1 = ((b - a) .* interval_X .+ b .+ a) ./ 2
w1 = weight * (b - a) / 2

a,b = [0,1]
interval_X, weight = gausslegendre(n)
x2 = ((b - a) .* interval_X .+ b .+ a) ./ 2
w2 = weight * (b - a) / 2

a,b = [0,1]
interval_X, weight = gausslegendre(n)
x3 = ((b - a) .* interval_X .+ b .+ a) ./ 2
w3 = weight * (b - a) / 2

z1 = x1
z2 = x2
z3 = x3

W = vcat([vec(w1*w2')*w3[i] for i=1:n]...)
W_mat = repeat(reshape(W, :, 1), 1, batchsize)

# Define hidden layer
hidden_channels = 40

# Load input of BRIDGE 
input = readdlm("train_BRIDGE/3d/matrix_Gz1.csv")
input = vcat(input, reshape(dm_list,1,batchsize))
input = vcat(input, reshape(λ_list,1,batchsize))
input = vcat(input, reshape(dp_list,1,batchsize))
input_list = [input[:,i] for i=1:batchsize]

# Load groud truth of BRIDGE 
train_sol = readdlm("train_BRIDGE/3d/matrix_Gz1z2z3.csv")
train_sol = reshape(train_sol,(length(z2)*length(z2)*length(z2),batchsize))

# Define BRIDGE model
model = Chain(Dense(length(z1)+3, hidden_channels,tanh),Dense(hidden_channels, length(z1)*length(z2)*length(z3)),x -> softplus.(x))
params, re = Flux.destructure(model);
ps = Flux.params(params);

# Define loss function
function loss_func(p)
    output = re(p).(input_list)
    output = hcat(output...)
    
    dist = (output.-train_sol).^2
    loss = mean(sum(dist.*W_mat,dims=1))
    return loss
end

# Training
lr_list = vcat([collect(0.01:-0.001:0.001) for i=1:10]...)

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
    end
end

# Write trained neural network parameters
using CSV,DataFrames
df = DataFrame(params = params)
CSV.write("train_BRIDGE/3d/params_trained.txt",df)


