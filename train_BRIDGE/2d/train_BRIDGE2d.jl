using Plots,Random,Distributions,Flux,DelimitedFiles,FastGaussQuadrature
using Flux,DelimitedFiles,Plots
using DataFrames,CSV
include("../../utils.jl")

# Read kinectic parameters for training
ps_true_matrix = readdlm("train_BRIDGE/2d/ps_for_trian.txt")
σon_list = ps_true_matrix[:,1]
σoff_list = ps_true_matrix[:,2]
ρ_list = ps_true_matrix[:,3]
dm_list = ps_true_matrix[:,4]
batchsize = length(σon_list)

# Define number of Gaussian Quadrature points 
N = length(z1)

# Define Gaussian Quadrature points and corresponding weights
n = N
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

X = [(x1[i],x2[j]) for i = 1 :length(x1) for j = 1:length(x2)]
W = vec(w1*w2')
W_mat = repeat(reshape(W, :, 1), 1, batchsize)

z1 = x1
z2 = x2

# Define hidden layer
hidden_channels = 40

# Load input of BRIDGE 
input = readdlm("train_BRIDGE/2d/matrix_Gz1.csv")
input = vcat(input, reshape(dm_list,1,batchsize))
input_list = [input[:,i] for i=1:batchsize]

# Load groud truth of BRIDGE 
train_sol = readdlm("train_BRIDGE/2d/matrix_Gz1z2.csv")

# Define BRIDGE model
model = Chain(Dense(length(z1)+1, hidden_channels,tanh),Dense(hidden_channels, length(z1)*length(z2)),x -> softplus.(x))
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
CSV.write("train_BRIDGE/2d/params_trained.txt",df)


t = Template(;
    user = "X-Y-Zhou",
    authors = ["X-Y-Zhou"],
    dir = "/Users/x-y-zhou/Documents/GitHub",
    julia = v"1.8",
    plugins = [
        License(; name = "MIT"),
        Git(; manifest = true),
        GitHubActions(),
        Documenter{GitHubActions}(),
        Readme(),
        Tests(),
    ],
)

t("BRIDGE")



