## Predictive Information Criteria

#function log_sum_exp(x)
#    xmax = maximum(x)
#    xmax + log(sum(exp.(x .- xmax)))
#end

AIC(x::T, data::Data) where T <: AbstractFit = AIC(merge(x.chains...), data)

## Akaike Information Criterion (AIC)
function AIC(x::T, data::Data) where T <: Chain
    map = MAP(x, data)
    N, q = size(x.theta[1].b)
    ## number of parameters without random effects and latent class indicators
    k = length(vectorize(map)) - N*q - N
    -2*(loglik(map, data) - k)
end

function mean_param(x::Fit{T}, data::Data) where T <: AbstractParam
    par = T(data, x.theta[1].G)
    for prop in fieldnames(typeof(par))
        y = mean([getproperty(w, prop) for w in x.theta])

        ## check for Int to avoid InexactError
        if eltype(getproperty(par, prop)) <: Int
            setproperty!(par, prop, Int.(round.(y, digits=0)))
        else
            setproperty!(par, prop, y)
        end
    end
    par
end

BIC(x::T, data::Data) where T <: AbstractFit = BIC(merge(x.chains...), data)

## Bayesian Information Criterion (BIC)
function BIC(x::T, data::Data) where T <: Chain
    post_mean = mean_param(x, data)
    
    N, q = size(x.theta[1].b)
    ## number of parameters without random effects and latent class indicators    
    k = length(vectorize(post_mean)) - N*q - N
    
    -2*loglik(post_mean, data) + k*log(N)
end

DIC(x::T, data::Data) where T <: AbstractFit = DIC(merge(x.chain...), data)

## Deviance Information Criterion (DIC)
function DIC(x::T, data::Data) where T <: Chain
    post_mean = mean_param(x, data)
    lppd = loglik(post_mean, data)
    pDIC = 2*(lppd - mean(map(par -> loglik(par, data), x.theta)))
    -2*(lppd - pDIC)
end

WAIC(x::T, data::Data) where T <: AbstractFit = WAIC(merge(x.chain...), data)

## Watanabe Akaike Information Criterion (WAIC)
function WAIC(x::T, data::Data) where T <: Chain
    N = length(data)
    pd = zeros(Float64, N)
    lpd = zeros(Float64, N)
    lpd_sq = zeros(Float64, N)
    
    M = n_samples(x)
    lp = get_loglik_p(x) ##dropdims(loglik_array(x, data), dims=(2))
    
    lppd = -N*log(M)
    for c in eachcol(lp) ##i in 1:N
        lppd += logsumexp(c) ##lp[:,i])
    end
    
    pWAIC1 = 2*(lppd - sum(mean(lp, dims=1)))
    pWAIC2 = sum(var(lp, dims=1))

    -2*(lppd - pWAIC1), -2*(lppd - pWAIC2)
end
    
function loo(x::T, data::Data; cores=1) where T <: AbstractFit
    ll = loglik_array(x, data)
    
    R"library(loo)"
    @rput ll
    @rput cores
    R"reff <- relative_eff(ll)"
    R"res <- loo(ll, r_eff=reff, cores=cores)"
    R"print(res)"
    R"w <- waic(ll)"
    R"print(w)"
end
