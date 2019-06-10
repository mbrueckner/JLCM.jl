convert(::Type{Chains}, x::T) where T <: Fit = get_mcmc_chains(x; include_subject_specific=false)

## convert Fit object to MCMCChains.Chains object    
function get_mcmc_chains(x::T; include_subject_specific=false) where T <: Fit
    get_mcmc_chains(x.chains; include_subject_specific=include_subject_specific)
end

function get_mcmc_chains(x::T; include_subject_specific=false) where T <: Chain
    get_mcmc_chains([x]; include_subject_specific=include_subject_specific)
end

## convert vector of Chain objects to MCMCChains.Chains object    
function get_mcmc_chains(x::AbstractVector{T}; include_subject_specific=false) where T <: Chain
    K = length(x)
    M = length(first(x))

    x0 = first(x).init
    N = length(vectorize(x0; include_subject_specific=include_subject_specific))
    ch = Array{Float64,3}(undef, M, N, K)

    for k in 1:K
        s = get_samples(x[k])
        for m in 1:M
            ch[m,:,k] = vectorize(s[m]; include_subject_specific=include_subject_specific)
        end
    end
    Chains(ch, variables(x0; include_subject_specific=include_subject_specific))
end
