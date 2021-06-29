## BEFORE RUNNING THE CODE BELOW, YOU MAY WANT TO RUN FormatingRotaetalData.jl

using BenchmarkTools
using StatsFuns
using SpecialFunctions
using Turing, DynamicHMC
using ReverseDiff
using MCMCChains, StatsPlots
using NNlib: softmax
using StatsFuns: logistic

# probabilistic model
@model multisppOccupancy(bob, coy, grayf, redf, X_detectionProb, Psicovariates, I_bobcat, I_coyote, I_grayFox, I_redFox) = begin

    # objects
    DetectHistoryProbBC = Array{Any}(undef, S) # (cd1 in Rota et al)
    DetectHistoryProbCO = Array{Any}(undef, S) # (cd2 in Rota et al)
    DetectHistoryProbGF = Array{Any}(undef, S) # (cd1 in Rota et al)
    DetectHistoryProbRF = Array{Any}(undef, S) # (cd2 in Rota et al)
    ψ = Array{Any}(undef, C, S)
    prob_ψs = Array{Any}(undef, S, C)
    z = Array{Any}(undef, S, C)

    # priors
    p_parametersBOB ~ filldist(Logistic(0, 1), 3) # Normal(0, 1) is the best so far
    p_parametersCO ~ filldist(Logistic(0, 1), 3)
    p_parametersGF ~ filldist(Logistic(0, 1), 3)
    p_parametersRF ~ filldist(Logistic(0, 1), 3)
    ψ_parameters ~ filldist(Logistic(0, 1), 32)

    # define detection parameter vectors
    DetecProbBCParat = Vector{Array{Any}}(undef, S)
    DetecProbBCParat[1:S] = repeat([p_parametersBOB], S)

    DetecProbCOParat = Vector{Array{Any}}(undef, S)
    DetecProbCOParat[1:S] = repeat([p_parametersCO], S)

    DetecProbGFParat = Vector{Array{Any}}(undef, S)
    DetecProbGFParat[1:S] = repeat([p_parametersGF], S)

    DetecProbRFParat = Vector{Array{Any}}(undef, S)
    DetecProbRFParat[1:S] = repeat([p_parametersRF], S)

        # detection probability at each replicate survey (lp in Rota et al)
        # note: at the logistic scale: (exp(η) / (1 + exp(η)))
        DetecProbBC = map.(logistic, X_detectionProb .* DetecProbBCParat)
        DetecProbCO = map.(logistic, X_detectionProb .* DetecProbCOParat)
        DetecProbGF = map.(logistic, X_detectionProb .* DetecProbGFParat)
        DetecProbRF = map.(logistic, X_detectionProb .* DetecProbRFParat)

        # f(z|ψ) probability of observing the detection history at each site (cd in Rota et al)
        # note: at the response scale
        for s in 1:S
        DetectHistoryProbBC[s] = exp(sum((bob[s] .* log.(DetecProbBC[s])) + ((1 .- bob[s]) .* log.(1 .- DetecProbBC[s]))))
        DetectHistoryProbCO[s] = exp(sum((coy[s] .* log.(DetecProbCO[s])) + ((1 .- coy[s]) .* log.(1 .- DetecProbCO[s]))))
        DetectHistoryProbGF[s] = exp(sum((grayf[s] .* log.(DetecProbGF[s])) + ((1 .- grayf[s]) .* log.(1 .- DetecProbGF[s]))))
        DetectHistoryProbRF[s] = exp(sum((redf[s] .* log.(DetecProbRF[s])) + ((1 .- redf[s]) .* log.(1 .- DetecProbRF[s]))))
        end

        # probability of each unique combination of 1s and 0s (psi in Rota et al)
        # there are 16 uique combinations: 1111 1110 1101 1100 1011 1010 1001 1000 0111 0110 0101 0100 0011 0010 0001 0000
        # there are 32 parameters linked to occupancy

        # # define occupancy parameter vectors
        PsiCOParat = Vector{Array{Any}}(undef, S)
        PsiCOParat[1:S] = repeat([ψ_parameters], S)

        # psi * probability of detection history (prob on Rota et al)
        for s in 1:S

        ψ[:, s] = softmax(Psicovariates[s, :, :] * ψ_parameters)

        prob_ψs[s, 1] = ψ[1, s] * DetectHistoryProbBC[s] * DetectHistoryProbCO[s] * DetectHistoryProbGF[s] * DetectHistoryProbRF[s] + 0.01e-300 # ψ_1111
        prob_ψs[s, 2] = ψ[2, s] * DetectHistoryProbBC[s] * DetectHistoryProbCO[s] * DetectHistoryProbGF[s] + 0.01e-300 # ψ_1110
        prob_ψs[s, 3] = ψ[3, s] * DetectHistoryProbBC[s] * DetectHistoryProbCO[s] * DetectHistoryProbRF[s] + 0.01e-300 # ψ_1101
        prob_ψs[s, 4] = ψ[4, s] * DetectHistoryProbBC[s] * DetectHistoryProbCO[s] + 0.01e-300 # ψ_1100
        prob_ψs[s, 5] = ψ[5, s] * DetectHistoryProbBC[s] * DetectHistoryProbGF[s] * DetectHistoryProbRF[s] + 0.01e-300 # ψ_1011
        prob_ψs[s, 6] = ψ[6, s] * DetectHistoryProbBC[s] * DetectHistoryProbGF[s] + 0.01e-300 # ψ_1010
        prob_ψs[s, 7] = ψ[7, s] * DetectHistoryProbBC[s] * DetectHistoryProbRF[s] + 0.01e-300 # ψ_1001
        prob_ψs[s, 8] = ψ[8, s] * DetectHistoryProbBC[s] + 0.01e-300 # ψ_1000
        prob_ψs[s, 9] = ψ[9, s] * DetectHistoryProbCO[s] * DetectHistoryProbGF[s] * DetectHistoryProbRF[s] + 0.01e-300 # ψ_0111
        prob_ψs[s, 10] = ψ[10, s] * DetectHistoryProbCO[s] * DetectHistoryProbGF[s] + 0.01e-300 # ψ_0110
        prob_ψs[s, 11] = ψ[11, s] * DetectHistoryProbCO[s] * DetectHistoryProbRF[s] + 0.01e-300 # ψ_0101
        prob_ψs[s, 12] = ψ[12, s] * DetectHistoryProbCO[s] + 0.01e-300 # ψ_0100
        prob_ψs[s, 13] = ψ[13, s] * DetectHistoryProbGF[s] * DetectHistoryProbRF[s] + 0.01e-300 # ψ_0011
        prob_ψs[s, 14] = ψ[14, s] * DetectHistoryProbGF[s] + 0.01e-300 # ψ_0010
        prob_ψs[s, 15] = ψ[15, s] * DetectHistoryProbRF[s] + 0.01e-300 # ψ_0001
        prob_ψs[s, 16] = ψ[16, s] + 0.01e-300 # ψ_0000

        z[s, 1] = I_bobcat[s] * I_coyote[s] * I_grayFox[s] * I_redFox[s] * log(prob_ψs[s, 1]) # ψ_1111
        z[s, 2] = I_bobcat[s] * I_coyote[s] * I_grayFox[s] * (1 - I_redFox[s]) * log(prob_ψs[s, 1] + prob_ψs[s, 2]) # ψ_1110
        z[s, 3] = I_bobcat[s] * I_coyote[s] * (1 - I_grayFox[s]) * I_redFox[s] * log(prob_ψs[s, 1] + prob_ψs[s, 3]) # ψ_1101
        z[s, 4] = I_bobcat[s] * I_coyote[s] * (1 - I_grayFox[s]) * (1 - I_redFox[s]) * log(prob_ψs[s, 1] + prob_ψs[s, 2] + prob_ψs[s, 3] + prob_ψs[s, 4]) # ψ_1100
        z[s, 5] = I_bobcat[s] * (1 - I_coyote[s]) * I_grayFox[s] * I_redFox[s] * log(prob_ψs[s, 1] + prob_ψs[s, 5]) # ψ_1011
        z[s, 6] = I_bobcat[s] * (1 - I_coyote[s]) * I_grayFox[s] * (1 - I_redFox[s]) * log(prob_ψs[s, 1] + prob_ψs[s, 2] + prob_ψs[s, 5] + prob_ψs[s, 6]) # ψ_1010
        z[s, 7] = I_bobcat[s] * (1 - I_coyote[s]) * (1 - I_grayFox[s]) * I_redFox[s] * log(prob_ψs[s, 1] + prob_ψs[s, 3] + prob_ψs[s, 5] + prob_ψs[s, 7]) # ψ_1001
        z[s, 8] = I_bobcat[s] * (1 - I_coyote[s]) * (1 - I_grayFox[s]) * (1 - I_redFox[s]) * log(prob_ψs[s, 1] + prob_ψs[s, 2] + prob_ψs[s, 3] + prob_ψs[s, 4] + prob_ψs[s, 5] +
        prob_ψs[s, 6] + prob_ψs[s, 7] + prob_ψs[s, 8]) # ψ_1000
        z[s, 9] = (1 - I_bobcat[s]) * I_coyote[s] * I_grayFox[s] * I_redFox[s] * log(prob_ψs[s, 1] + prob_ψs[s, 9]) # ψ_0111
        z[s, 10] = (1 - I_bobcat[s]) * I_coyote[s] * I_grayFox[s] * (1 - I_redFox[s]) * log(prob_ψs[s, 1] + prob_ψs[s, 2] + prob_ψs[s, 9] + prob_ψs[s, 10]) # ψ_0110
        z[s, 11] = (1 - I_bobcat[s]) * I_coyote[s] * (1- I_grayFox[s]) * I_redFox[s] * log(prob_ψs[s, 1] + prob_ψs[s, 3] + prob_ψs[s, 9] + prob_ψs[s, 11]) # ψ_0101
        z[s, 12] = (1 - I_bobcat[s]) * I_coyote[s] * (1 - I_grayFox[s]) * (1 - I_redFox[s]) * log(prob_ψs[s, 1] + prob_ψs[s, 2] + prob_ψs[s, 3] + prob_ψs[s, 4] + prob_ψs[s, 9] +
        prob_ψs[s, 10] + prob_ψs[s, 11] + prob_ψs[s, 12]) # ψ_0100
        z[s, 13] = (1 - I_bobcat[s]) * (1 - I_coyote[s]) * I_grayFox[s] * I_redFox[s] * log(prob_ψs[s, 1] + prob_ψs[s, 5] + prob_ψs[s, 9] + prob_ψs[s, 13]) # ψ_0011
        z[s, 14] = (1 - I_bobcat[s]) * (1 - I_coyote[s]) * I_grayFox[s] * (1 - I_redFox[s]) * log(prob_ψs[s, 1] + prob_ψs[s, 2] + prob_ψs[s, 5] + prob_ψs[s, 6] + prob_ψs[s, 9] +
        prob_ψs[s, 10] + prob_ψs[s, 13] + prob_ψs[s, 14]) # ψ_0010
        z[s, 15] = (1 - I_bobcat[s]) * (1 - I_coyote[s]) * (1 - I_grayFox[s]) * I_redFox[s] * log(prob_ψs[s, 1] + prob_ψs[s, 3] + prob_ψs[s, 5] + prob_ψs[s, 7] + prob_ψs[s, 9] +
        prob_ψs[s, 11] + prob_ψs[s, 13] + prob_ψs[s, 15]) # ψ_0001
        z[s, 16] = (1 - I_bobcat[s]) * (1 - I_coyote[s]) * (1 - I_grayFox[s]) * (1 - I_redFox[s]) * log(sum(prob_ψs[s, :])) # ψ_0000

        Turing.@addlogprob! sum(z[s, :])

       end
    end
# Settings of the Hamiltonian Monte Carlo (HMC) sampler
iterations = 2000
ϵ = 0.01
τ = 5
outcome = mapreduce(c -> sample(multisppOccupancy(bobcatBM, coyoteBM, grayFoxBM, redFoxBM, XX_detectionProb, X, I_bobcat, I_coyote, I_grayFox, I_redFox),
Gibbs(
    HMC{Turing.ForwardDiffAD{12}}(ϵ, τ, :p_parametersBOB,:p_parametersCO, :p_parametersGF,:p_parametersRF),
    HMC{Turing.TrackerAD}(ϵ, τ, :ψ_parameters)
), iterations, drop_warmup = false, progress = true, verbose = true), chainscat, 1:2)
