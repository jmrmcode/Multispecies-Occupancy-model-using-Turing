using DelimitedFiles
using StatsBase

# Data files from Rota, Christopher T. et al. (2017), Data from: A multispecies occupancy model for two or more interacting species, Dryad, Dataset, https://doi.org/10.5061/dryad.pq624

# import detection histories
# keep in mind that when processing the carnivore detection I´ll have to discard "NA"
bobcat = convert(Array{Union{Int64, SubString{String}}, 2}, readdlm("D:/EMERGIA/code/Rota et al 2016/Data/Bobcat.csv", ','))[1:end, 1:end]
coyote = convert(Array{Union{Int64, SubString{String}}, 2}, readdlm("D:/EMERGIA/code/Rota et al 2016/Data/Coyote.csv", ','))[1:end, 1:end]
grayFox = convert(Array{Union{Int64, SubString{String}}, 2}, readdlm("D:/EMERGIA/code/Rota et al 2016/Data/Gray Fox.csv", ','))[1:end, 1:end]
redFox = convert(Array{Union{Int64, SubString{String}}, 2}, readdlm("D:/EMERGIA/code/Rota et al 2016/Data/Red Fox.csv", ','))[1:end, 1:end]

# import detection (p) covariates
# wheter or not a camera was on or off in a trail (trl) and the total detection distance of the camera (dd)
p_covariate = convert(Array{Float64, 2}, readdlm("D:/EMERGIA/code/Rota et al 2016/Data/p covariates.csv", ',')[2:end, 1:end]) # [2:end, 1:end]
p_covariateS = standardize(ZScoreTransform, p_covariate[:, 1]; dims = 1)

# import occupancy (ψ) covariates
psi_covariates = convert(Array{Float64, 2}, readdlm("D:/EMERGIA/code/Rota et al 2016/Data/psi covariates.csv", ',')[2:end, 1:end])

# sites, surveys and occupancy unique combinations
S = size(bobcat)[1]  # number of sites
J = Array{Int8, 2}(undef, S, 1) # number of camera trapping days at each site
# populate J. Note that the number of camera trapping days per site is the same across the species, but NAs are different across sites
for i in 1:S
    J[i] = length(filter!(e -> e ≠ "NA", bobcat[i, :]))
end
C = 16 # number of unique combination of 1s and 0s

# create block matrices for each species
bobcatBM = Vector{Array{Int64}}(undef, S)
for s in 1:S
    bobcatBM[s] = convert(Array{Int64,1}, bobcat[s, findall(x->x≠"NA", bobcat[s, :])])
end

coyoteBM = Vector{Array{Float64}}(undef, S)
for s in 1:S
    coyoteBM[s] = convert(Array{Real,1}, coyote[s, findall(x->x≠"NA", coyote[s, :])])
end

grayFoxBM = Vector{Array{Float64}}(undef, S)
for s in 1:S
    grayFoxBM[s] = convert(Array{Real,1}, grayFox[s, findall(x->x≠"NA", grayFox[s, :])])
end

redFoxBM = Vector{Array{Float64}}(undef, S)
for s in 1:S
    redFoxBM[s] = convert(Array{Real,1}, redFox[s, findall(x->x≠"NA", redFox[s, :])])
end

# create a block matrix having the covariates for detection probability
XX_detectionProb = Vector{Array{Float64}}(undef, S)
for s in 1:S
    XX_detectionProb[s] = [repeat([1], J[s]) repeat([psi_covariates[s, 6]], J[s]) repeat([p_covariate[s]], J[s])]
end

# standarization of occupancy covariates
# StatsBase.fit(X::Array{Any, 2}) = StatsBase.fit()
# Zpsi_covariates = StatsBase.fit(ZScoreTransform, psi_covariates[:, 1:4]; dims = 1)
Zpsi_covariates = standardize(ZScoreTransform, psi_covariates[:, 1:4]; dims = 1) # d5km, hden, lati, long
latxlon = Zpsi_covariates[:, 3] .* Zpsi_covariates[:, 4] # lbyl
hike = (psi_covariates[:, 5] .* 1000) ./ J
hike = standardize(ZScoreTransform, hike[:, 1]) # hike

# import Design Matrix
dm = convert(Array{Any, 2}, readdlm("D:/EMERGIA/code/Rota et al 2016/Data/Design Matrix.csv", ',')[3:end, 2:end])

# design matrix for unique combinations of ψs
X = Array{Float64}(undef, S, C, 32)
# populate x (adapted from Formatting Data.R in Rota et al's appendix). This is X.array
for i in 1:S
    for j in 1:(C-1) # discard ψ_0000
        X[i, j, findall(!iszero, dm[j, :])] =
          [1, Zpsi_covariates[i, 3], Zpsi_covariates[i, 4], latxlon[i], hike[i], #f1
            1, Zpsi_covariates[i, 3], Zpsi_covariates[i, 4], latxlon[i], Zpsi_covariates[i, 2], #f2
            1, Zpsi_covariates[i, 3], Zpsi_covariates[i, 4], latxlon[i], Zpsi_covariates[i, 2], #f3
            1, Zpsi_covariates[i, 3], Zpsi_covariates[i, 4], latxlon[i], hike[i], #f4
            1, Zpsi_covariates[i, 1], #f12
            1, Zpsi_covariates[i, 2], #f13
            1, Zpsi_covariates[i, 2], #f14
            1, Zpsi_covariates[i, 1], #f23
            1, Zpsi_covariates[i, 2], #f24
            1, Zpsi_covariates[i, 1]][findall(!iszero, dm[j, :])] #f34
    end
end
# populate the block matrix used as an input. X above is an subproduct
X2 = Vector{Array{Any}}(undef, S)
for s in 1:S
    X2[s] = X[s,:,:]
end

# I_Sp_bobcat, I_coyote, I_grayFox, and I_redFox are vectors of length the number of sites (S) that contains whether (1) or not (0)
# the species were detected at each site
I_bobcat = Array{Int64}(undef, S)
I_coyote = Array{Int64}(undef, S)
I_grayFox = Array{Int64}(undef, S)
I_redFox = Array{Int64}(undef, S)

# populate I_Sp_bobcat
# accumulated detections across sites
sumDetectBOB = zeros(S)
for s in 1:S
    sumDetectBOB[s] = sum(bobcat[s, findall(x->x≠"NA", bobcat[s, :])])
end
for s in 1:S
    I_bobcat[s] = ifelse(sumDetectBOB[s] > 0.0, 1, 0)
end

# populate I_coyote
# accumulated detections across sites
sumDetectCO = zeros(S)
for s in 1:S
    sumDetectCO[s] = sum(coyote[s, findall(x->x≠"NA", coyote[s, :])])
end
for s in 1:S
    I_coyote[s] = ifelse(sumDetectCO[s] > 0.0, 1, 0)
end

# populate I_grayFox
# accumulated detections across sites
sumDetectGF = zeros(S)
for s in 1:S
    sumDetectGF[s] = sum(grayFox[s, findall(x->x≠"NA", grayFox[s, :])])
end
for s in 1:S
    I_grayFox[s] = ifelse(sumDetectGF[s] > 0.0, 1, 0)
end

# populate I_redFox
# accumulated detections across sites
sumDetectRF = zeros(S)
for s in 1:S
    sumDetectRF[s] = sum(redFox[s, findall(x->x≠"NA", redFox[s, :])])
end
for s in 1:S
    I_redFox[s] = ifelse(sumDetectRF[s] > 0.0, 1, 0)
end
