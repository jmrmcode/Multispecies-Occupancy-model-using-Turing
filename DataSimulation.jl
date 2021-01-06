using Random
using Distributions
using StatsFuns: logistic

###**** DATA SIMULATION (for two species)*****######

Random.seed!(2345)
S = 100  # number of sites
J = 10 # number of surveys at each site
C = 4 # number of unique combination of 1s and 0s

## linear predictor for marginal Sp1 occupancy
# continuous predictor
x1occpancy = rand(Normal(0, 1), S)
# matrix X1
X1occupancy = Array{Float64}(undef, S, 2)
# populate matrix X1
for i in 1:S
        X1occupancy[i, :] = vcat(1, x1occpancy[i])
    end
# coeffcients vector (model parameters)
β1occupancy = vcat(0, 2)
# probability of presence (occupancy)
ψ_10 = logistic.(X1occupancy * β1occupancy)

# detection history
z_10 = Array{Bool}(undef, S, 1)
for i in 1:S
    z_10[i] = rand(Bernoulli(ψ_10[i]), i)[1]
end

## linear predictor for marginal Sp1 detection
# continuous predictor
x1detection = rand(Normal(0, 2), S)
# matrix X2
X1detection = Array{Float64}(undef, S, 2)
# populate matrix X2
for i in 1:S
        X1detection[i, :] = vcat(1, x1detection[i])
    end
# coeffcients vector (model parameters)
β1detection = vcat(0, -1)
# probability of presence (occupancy)
p_Sp1 = logistic.(X1detection * β1detection)

# detection history
Sp1Detections = Array{Real}(undef, S, J) # declare data structure
# populate detection history
for s in 1:S
    for j in 1:J
    Sp1Detections[s, j] = rand(Bernoulli(z_10[s] * p_Sp1[s]), 1)[1]
    end
end

# accumulated detections across sites
sumDetectSp1 = zeros(S)
for i in 1:length(Sp1Detections[:, 1])
    sumDetectSp1[i] = sum(Sp1Detections[i, :])
end

## linear predictor for marginal Sp2 occupancy
# continuous predictor
x2occpancy = rand(Normal(0, 1.5), S)
# matrix X1
X2occupancy = Array{Float64}(undef, S, 2)
# populate matrix X1
for i in 1:S
        X2occupancy[i, :] = vcat(1, x2occpancy[i])
    end
# coeffcients vector (model parameters)
β2occupancy = vcat(0, -2)
# probability of presence (occupancy)
ψ_01 = logistic.(X2occupancy * β2occupancy)

# detection history
z_01 = Array{Bool}(undef, S, 1)
for i in 1:S
    z_01[i] = rand(Bernoulli(ψ_01[i]), i)[1]
end

## linear predictor for marginal Sp2 detection
# continuous predictor
x2detection = rand(Normal(0, 3), S)
# matrix X2
X2detection = Array{Float64}(undef, S, 2)
# populate matrix X2
for i in 1:S
        X2detection[i, :] = vcat(1, x2detection[i])
    end
# coeffcients vector (model parameters)
β2detection = vcat(0, 2)
# probability of presence (occupancy)
p_Sp2 = logistic.(X2detection * β2detection)

# detection history
Sp2Detections = Array{Real}(undef, S, J) # declare data structure
# populate detection history
for s in 1:S
    for j in 1:J
    Sp2Detections[s, j] = rand(Bernoulli(z_01[s] * p_Sp2[s]), 1)[1]
    end
end

# accumulated detections across sites
sumDetectSp2 = zeros(S)
for i in 1:length(Sp1Detections[:, 1])
    sumDetectSp2[i] = sum(Sp2Detections[i, :])
end

#####**** COMPILE THE INPUTS REQUIRED BY THE MODEL *****###################
# x1detc and x2detc
x1detc = Array{Float64}(undef, S, J)
x2detc = Array{Float64}(undef, S, J)
# populate x1detc and x2detc, note that the covariates value is the same across surveys at each site
for i in 1:S
        x1detc[i, :] = repeat([x1detection[i]], J)
        x2detc[i, :] = repeat([x2detection[i]], J)
    end

# x is a matrix containing the model matrix for the unique combinations of 1s and 0s (x in Rota et al)
# 5 columns = intercept for f1, covariate for f1, intercept for f2, covariate for f2, and intercept for f12
# rows at each site corresponds to ψ_11, ψ_10, ψ_01, and ψ_00
# x matrix says which fs are involved in each ψ
x = Array{Float64}(undef, S, C, 5)
# populate x; this is X.array in Formatting Data.R, Rota et al's appendix
for i in 1:S
    x[i,:,:] = [1 x1occpancy[i] 1 x2occpancy[i] 1;1 x1occpancy[i] 0 0 0;0 0 1 x2occpancy[i] 0;0 0 0 0 0]
end

# I_Sp1 and I_Sp2 are vectors of length the number of sites (S) that contains whether (1) or not (0)
# each species was detected at each site
I_Sp1 = Array{Int64}(undef, S)
I_Sp2 = Array{Int64}(undef, S)
# populate I_Sp1
for s in 1:S
    I_Sp1[s] = ifelse(sumDetectSp1[s] > 0.0, 1, 0)
end
# populate I_Sp2
for s in 1:S
    I_Sp2[s] = ifelse(sumDetectSp2[s] > 0.0, 1, 0)
end
