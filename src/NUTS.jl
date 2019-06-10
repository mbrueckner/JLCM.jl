module NUTSGibbs

using LinearAlgebra, Statistics
using TransformVariables, LogDensityProblems, DynamicHMC, ForwardDiff, Parameters
using Random: Random

import DynamicHMC: NUTS_init, NUTS_transition, get_final_ϵ, adapting_ϵ, adapt_stepsize
import LogDensityProblems: ADgradient

abstract type AbstractProblem end

export NUTSGibbsSampler

mutable struct NUTSGibbsSampler{Tp<:AbstractProblem, Tv, Tf, TA, TApar, Tfunc<:Function}
    iter::Int
    warmup::Int
    nuts::NUTS{Tv,Tf}
    problem::Tp
    transformation::Tfunc
    eps::Float64
    A_params::TApar
    A::TA
    sample::Vector{NUTS_Transition{Tv,Tf}}
    tune_steps::Vector{Int}
end

function NUTSGibbsSampler(nuts::NUTS{Tv,Tf}, problem, transformation, eps, warmup, A_params, A) where {Tv,Tf}
    sample = Vector{NUTS_Transition{Tv,Tf}}(undef, 0)
    tune_steps = floor.(Int, [0.15, 0.2, 0.3, 0.5, 0.9].*warmup)
    @info tune_steps
    NUTSGibbsSampler(0, warmup, nuts, problem, transformation, eps, A_params, A, sample, tune_steps)
end
    
function init(problem::Tp, transformation::Tf, warmup::Int) where {Tp <: AbstractProblem, Tf <: Function}
    P = TransformedLogDensity(transformation(problem), problem)
    nuts = nuts_try_init(ADgradient(:ForwardDiff, P))
    A_params, A = adapting_ϵ(nuts.ϵ)
    NUTSGibbsSampler(nuts, problem, transformation, nuts.ϵ, warmup, A_params, A)
end

function nuts_try_init(P)
    try
        nuts = NUTS_init(Random.GLOBAL_RNG, P; report=ReportSilent())
        @info "NUTS initial stepsize eps=$(nuts.ϵ)!"
        return nuts
    catch ex
        nuts = NUTS_init(Random.GLOBAL_RNG, ϵ=0.1, P)
        @warn "NUTS using fixed step size (eps=$(nuts.ϵ))!", ex
        return nuts
    end
end

function tune_cov(nuts::NUTS, A, sample; regularize=5.0)
    @unpack rng, H, max_depth, report = nuts

    Σ = DynamicHMC.sample_cov(sample)
    Σ += (DynamicHMC.UniformScaling(max(1e-3, median(diag(Σ))))-Σ) * regularize/length(sample)
    κ = DynamicHMC.GaussianKE(Symmetric(Σ))
    
    NUTS(rng, DynamicHMC.Hamiltonian(H.ℓ, κ), sample[end].q, get_final_ϵ(A), max_depth, report)
end

function step(x::NUTSGibbsSampler, current_pos)
    @unpack rng, H, q, ϵ, max_depth, report = x.nuts

    res = current_pos
    ac = 0.0
    x.iter += 1
    
    try
        pos = first(inverse.(Ref(x.transformation(x.problem)), [current_pos]))
        trans = NUTS_transition(rng, H, pos, x.eps, max_depth)
        res = first(transform.(Ref(x.transformation(x.problem)), [get_position(trans)]))
        ac = get_acceptance_rate(trans)
        
        if x.iter <= x.warmup
            x.A = adapt_stepsize(x.A_params, x.A, trans.a)
            x.eps = get_final_ϵ(x.A)

            push!(x.sample, trans)
            
            if x.iter ∈ x.tune_steps ##[100, 150, 250, 450, 850]
                x.nuts = tune_cov(x.nuts, x.A, x.sample)
                resize!(x.sample, 0)
            end

            if x.iter == x.warmup
                @info "Finish adaptation"
                @info "Final stepsize: $(x.eps)"
                @info "Mean acceptance rate at current transition: $(ac)"
            end
        end        
    catch ex
        @warn "Exception in NUTS_transition: ", ex
    end
    
    return res, ac
end

end
