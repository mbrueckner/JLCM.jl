## estimate MAP by first estimating MAP for each chain
MAP(x::T, data) where T <: Fit = MAP([MAP(ch, data) for ch in x.chains], data)

MAP(x::Chain, data) = x.theta[last(findmax(sum(x.loglik_p, dims=2)[:,1]))]

function MAP(theta::Vector{T}, data) where T <: AbstractParam
    M = length(theta)
    ll = -Inf
    tmax = 1
    
    for t in 1:M
        new_ll = loglik(theta[t], data)

        if new_ll > ll
            ll = new_ll
            tmax = t
        end
    end

    theta[tmax]
end

weibull_loghaz(time, shape, scale) = log(scale) .+ log(shape) .+ (shape-1) .* log.(time)
weibull_cumhaz(time, shape, scale) = scale .* time.^shape
##inv_weibull_cumhaz(p, shape, scale) = (p ./ scale).^(1/shape)

function loglik_p_surv(T, E, Xe, shape::Tx, scale::Tx, delta::Vector{Tx}) where Tx <: Real
    if (shape .<= 0.0) | (scale .<= 0.0)
        -Inf
    else
        Xed = Xe*delta
        sum(E.*(weibull_loghaz(T .- 1, shape, scale) .+ Xed) .- weibull_cumhaz(T .- 1, shape, scale).*exp.(Xed))
    end
end

function loglik_p_surv(T, E, Xe, haz_basis, cumhaz_basis, w, delta)
    Xed = Xe*delta
    ew = exp.(w)
    sum(E.*(log.(haz_basis*ew) .+ Xed) .- cumhaz_basis*ew.*exp.(Xed))
end

##invquad(L, x) = dot(x, L' \ (L \ x))
##mvn_logpdf(x, mu, L) = -length(mu)/2 * log(2*pi) - log(det(L)) - invquad(L, x-mu)/2

function loglik_p_long(b, beta, mu, Sigma, sigma, Y, times, n, Xl, Z)
    target = zero(eltype(mu))
    d = Normal(0, sigma)
    
    for j in 1:n
        target += logpdf(d, Y[j] - dot(Z[j,:], b) - dot(Xl[j,:], beta))        
        ##target += logpdf(Normal(zero(eltype(sigma)), sigma), Y[j] - dot(Z[j,:], u) - dot(Xl[j,:], beta))
        ## loglikelihood(Normal(0, sigma), Y[1:n] .- Z[1:n,:]*u - Xl[1:n,:]*beta)
    end

    target + logpdf(MvNormal(mu, Sigma), b) ##mvn_logpdf(b, zero(b), Sigma)
end

function loglik_p_long(beta, mu, Sigma, sigma, Y, times, n, Xl, Z)
    m = Z*mu + Xl*beta    
    C = Array(Symmetric(Z*Sigma*Z' + sigma^2*diagm(0 => ones(Float64, n))))
    issuccess(cholesky(C; check=false)) || return -Inf
    logpdf(MvNormal(m, C), Y)
end

function loglik_p(theta::Param, data)
    @unpack T, E, Xe, Xl, Xp, Z, Y, times, n = data
    @unpack xi, cind, delta, bl_shape, bl_scale, b, beta, mu, S, L, sigma = theta
    
    cp = class_prob(xi, Xp)
    
    N = length(T)
    G = size(beta)[1]
    
    loglik = zeros(Float64, N)

    Sigma = [build_cov_matrix(S[g,:], L[g,:,:]) for g in 1:G]
    
    for i in 1:N
        g = cind[i]
        lll_vec = loglik_p_long(b[i,:], beta[g,:], mu[g,:], Sigma[g], sigma, Y[i,1:n[i]], times[i,1:n[i]], n[i], Xl[i,1:n[i],:], Z[i,1:n[i],:])
        lle_vec = loglik_p_surv(T[i], E[i], Xe[i,:]', bl_shape[g], bl_scale[g], delta[g,:])
        loglik[i] = lll_vec + lle_vec + log(cp[i,g])
    end

    loglik
end

function loglik_p(theta::ParamSpline, data)
    @unpack T, E, Xe, Xl, Xp, Z, Y, times, n, haz_basis, cumhaz_basis = data
    @unpack xi, cind, delta, w, b, beta, mu, S, L, sigma = theta
    
    cp = class_prob(xi, Xp)
    
    N = length(T)
    G = size(beta)[1]
    
    ll = zeros(Float64, N)

    Sigma = [build_cov_matrix(S[g,:], L[g,:,:]) for g in 1:G]
    
    for i in 1:N
        g = cind[i]
        lll_vec = loglik_p_long(b[i,:], beta[g,:], mu[g,:], Sigma[g], sigma, Y[i,1:n[i]], times[i,1:n[i]], n[i], Xl[i,1:n[i],:], Z[i,1:n[i],:])
        lle_vec = loglik_p_surv(T[i], E[i], Xe[i,:]', haz_basis[i,:]', cumhaz_basis[i,:]', w[g,:], delta[g,:])
        ll[i] = lll_vec + lle_vec + log(cp[i,g])
    end
    ll
end

loglik(theta, data) = sum(loglik_p(theta, data))

loglik_array(x::T, data::Data) where T <: AbstractFit = loglik_array(x.chains, data)

function loglik_array(x::AbstractVector{T}, data::Data) where T <: Chain
    M, K, N = length(x[1].theta), length(x), length(data.T)
    
    ##lp = SharedArray{Float64,3}((M, K, N))
    lp = Array{Float64,3}(undef, M, K, N) ##SharedArray{Float64,3}((M, K, N))
    
    ##@sync @distributed
    for m in 1:M
        for k in 1:K
            lp[m,k,:] = loglik_p(x[k].theta[m], data)
        end
    end
    lp
end
