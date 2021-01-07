using Turing, DynamicHMC
using MCMCChains, StatsPlots
using NNlib: softmax

################***** Declare Multispecies Occupancy Model *******###########
@model multisppOccupancy(x1detc, x2detc, Sp1Detections, Sp2Detections, x, I_Sp1, I_Sp2, σ2) = begin
    
    # objects
    DetecProbSp1 = Array{Float64}(undef, J) # (lp1 in Rota et al)
    DetecProbSp2 = Array{Float64}(undef, J) # (lp2 in Rota et al)
    DetectHistoryProbSp1 = Array{Any}(undef, S) # (cd1 in Rota et al)
    DetectHistoryProbSp2 = Array{Any}(undef, S) # (cd2 in Rota et al)
    ψ = Array{Any}(undef, C, S) # (psi in Rota et al)
    jointOccupCov = Array{Float64}(undef, 5)
    prob_ψs = Array{Any}(undef, S, C)
    z = Array{Any}(undef, S, C)
    
    # priors
    interceptDetectSp1 ~ Logistic(0, 1)
    interceptDetectSp2 ~ Logistic(0, 1)
    covariateDetectSp1 ~ Normal(0, σ2)
    covariateDetectSp2 ~ Normal(0, σ2)
    α0 ~ Logistic(0, 1)
    α1 ~ Normal(0, 1)
    β0 ~ Logistic(0, 1)
    β1 ~ Normal(0, σ2)
    γ0 ~ Normal(0, σ2)
    jointOccupCov = [α0; α1; β0; β1; γ0]

    # likelihood
    for s in S # (loop over sites)
        # detection probability at each replicate survey (lp in Rota et al)
        DetecProbSp1 = logistic.(interceptDetectSp1 .+ covariateDetectSp1 .* x1detc[s, :])
        DetecProbSp2 = logistic.(interceptDetectSp2 .+ covariateDetectSp2 .* x2detc[s, :])
        
        # f(z|ψ) probability of observing the detection history at each site (cd in Rota et al)
        # Sp1Detections = y1 (in Rota et al)
        DetectHistoryProbSp1[s] = exp(sum(Sp1Detections[s, :] .* log.(DetecProbSp1) + (1 .- Sp1Detections[s, :]) .* log.(1 .- DetecProbSp1)))
        DetectHistoryProbSp2[s] = exp(sum(Sp2Detections[s, :] .* log.(DetecProbSp2) + (1 .- Sp2Detections[s, :]) .* log.(1 .- DetecProbSp2)))
        
        # probability of each unique combination of 1s and 0s (psi in Rota et al)
        # there are 4 uique combinations: 11 10 01 and 00
        # Eg: x[s, :, :] * jointOccupCov = α0 + α1*x1 + β0 + β1*x2 + γ0 = f1 + f2 + f12
        # Eg: ψ_11 = exp(f1 + f2 + f12) / 1 + exp(f1) + exp(f2) + exp(f1 + f1 + f12))
        ψ[:, s] = softmax(x[s, :, :] * jointOccupCov)
        
        # psi * probability of detection history (prob on Rota et al)
        prob_ψs[s, 1] = ψ[1, s] * DetectHistoryProbSp1[s] * DetectHistoryProbSp2[s] # ψ_11
        prob_ψs[s, 2] = ψ[2, s] * DetectHistoryProbSp1[s] # ψ_10
        prob_ψs[s, 3] = ψ[3, s] * DetectHistoryProbSp2[s] # ψ_01
        prob_ψs[s, 4] = ψ[4, s] # ψ_00
        
        # log contribution of each site to the likelihood (the log contribution of each unique combination at each site)
        # z in Rota et al
        z[s, 1] = I_Sp1[s] * I_Sp2[s] * log(prob_ψs[s, 1]) # ψ_11
        z[s, 2] = I_Sp1[s] * (1 - I_Sp2[s]) * log(prob_ψs[s, 1] + prob_ψs[s, 2]) # ψ_10
        z[s, 3] = (1 - I_Sp1[s]) * I_Sp2[s] * log(prob_ψs[s, 1] + prob_ψs[s, 3]) # ψ_01
        z[s, 4] = (1 - I_Sp1[s]) * (1 - I_Sp2[s]) * log(sum(prob_ψs[s, :])) # ψ_00
        
        # Update the accumulated log probability
        Turing.@addlogprob! sum(z[s, :])
        end
    end

# Start the No-U-Turn Sampler (NUTS)
outcome = mapreduce(c -> sample(multisppOccupancy(x1detc, x2detc, Sp1Detections, Sp2Detections, x, I_Sp1, I_Sp2, 3),
NUTS(1000, .95), 1000, drop_warmup = false), chainscat, 1:3)

# print outcome
display(outcome)
