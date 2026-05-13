using StatsBase,Distributions,DelimitedFiles
using CSV,DataFrames

# Convert a vector to probability distributions
function convert_histo(data::Vector)
    # Define histogram edge set (integers)
    max_np = ceil(maximum(data))+1
    min_np = 0
    edge = collect(min_np:1:max_np)
    H = fit(Histogram,data,edge)
    saved=zeros(length(H.weights),2);
    saved[:,1] = edge[1:end-1];
    # Normalize histogram to probability (since bins are defined on integers)
    saved[:,2] = H.weights/length(data);
    return saved[:,1], saved[:,2]
end

# Embeding the distribution to length N 
function embeding_dist(dist,N)
    if length(dist)<N
        dist = vcat(dist,zeros(N-length(dist)))
    else
        dist = dist[1:N]
    end
    return dist
end

# Calculate mean value according to the distribution P
P2mean(P) = [P[i] * (i-1) for i in 1:length(P)] |> sum

# Calculate variance var
P2var(P) = ([P[i] * (i-1)^2 for i in 1:length(P)] |> sum) - P2mean(P)^2

# Calculate second moment sm
P2sm(P) = [P[i] * (i-1)^2 for i in 1:length(P)] |> sum

# Normalization
function set_one(vec)
    vec = abs.(vec)
    vec = vec./sum(vec)
    return vec
end

# Convert distribution to PGF for 1d 2d and 3d
function hist_gf1d(hist_data,z)
    N = length(hist_data)
    z_vec = [z.^i for i = 0:N-1]
    return sum(z_vec.*hist_data)
end

function hist_gf2d(hist_data,z1,z2)
    Nx = size(hist_data,1)
    Ny = size(hist_data,2)
    z1_vec = [z1.^i for i = 0 : Nx-1]
    z2_vec = [z2.^i for i = 0 : Ny-1]
    z_mat = z1_vec*z2_vec'
    return sum(z_mat.*hist_data)
end

function hist_gf3d(hist_data,z1,z2,z3)
    Nx = size(hist_data,1)
    Ny = size(hist_data,2)
    Nz = size(hist_data,3)
    return [sum([hist_data[i,j,k]*z1_value^(i-1)*z2_value^(j-1)*z3_value^(k-1) for i=1:Nx for j=1:Ny for k=1:Nz]) 
    for z1_value in z1, z2_value in z2, z3_value in z3] 
end
