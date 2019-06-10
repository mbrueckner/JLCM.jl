struct Control
    iter::Int ## total number of iterations (incl. warmup)
    warmup::Int ## number of warmup iterations    
    chains::Int ## number of chains
    save_warmup::Bool ## store and return warmup iterations in Fit
    common_delta::Bool ## common log-HRs across classes
    common_beta::Bool ## common fixed effects across classes
    adapt_delta::Float64 ## target mean acceptance rate for NUTS transitions
end

Control(iter, chains) = Control(iter=iter, chains=chains)

function Control(;iter=2000, warmup=Int(floor(iter/2)), chains=4, save_warmup=false,
                 common_delta=true, common_beta=true, adapt_delta=0.8)
    @argcheck iter > warmup
    Control(iter, Int(floor(iter/2)), chains, save_warmup, common_delta, common_beta, adapt_delta)
end

struct Chain{Tpar<:AbstractParam, Tprior<:AbstractPrior}
    theta::Vector{Tpar}
    accept::Tpar
    loglik_p::Matrix{Float64} ## pointwise log-likelihood        
    prior::Tprior
    init::Tpar
end

struct Fit{Tpar<:AbstractParam, Tprior<:AbstractPrior} <: AbstractFit
    control::Control
    chains::Vector{Chain{Tpar, Tprior}}
end

function show(io::IO, x::Data)
    println(io, "Subjects: $(length(x.T))")
    println(io, "Repeated measurements: $(sum(x.n))")
    println(io, "Events: $(sum(x.E))")
end
    
function show(io::IO, x::Chain{T,S}) where {T,S}
    cm = countmap([par.G for par in x.theta])
    cm0 = countmap([par.G-par.G0 for par in x.theta])

    println(io, "Number of groups: $(collect(zip(keys(cm), values(cm))))")
    println(io, "Number of non-empty groups: $(collect(zip(keys(cm0), values(cm0))))")
end

function show(io::IO, x::Fit{T}) where T <: AbstractParam
    println(io, "Number of chains: $(x.control.chains)")
    println(io, "Iterations: $(x.control.iter)")
    for i in eachindex(x.chains)
        println(io, "Chain $(i):")
        show(io, x.chains[i])
        println(io, "\n")
    end
end

merge(a...) = foldr(merge, a)
merge(a::Chain{T,S}) where {T,S} = a

function merge(a::Chain{T,S}, b::Chain{T,S}) where {T <: AbstractParam, S <: AbstractPrior}
    Chain{T,S}(vcat(a.theta, b.theta), a.accept, vcat(a.loglik_p, b.loglik_p), a.prior, a.init)
end

size(x::Fit) = (length(x.chains), length(first(x.chains)))
slice(x::Chain, t) = x.theta[t]

length(x::Fit) = length(x.chains)
length(x::Chain) = length(x.theta)

function get_samples(x::Fit)
    K, M = size(x)
    res = Matrix{typeof(first(x.chains).init)}(undef, M, K)

    for k in 1:K
        res[:,k] = get_samples(x.chains[k])
    end
    res
end
        
get_samples(x::Chain) = x.theta

get_loglik_p(x::Chain) = x.loglik_p
get_loglik_p(x::Fit) = get_loglik_p(merge(x.chains...))

get_acceptance_rates(x::Fit) = get_acceptance_rates(merge(x.chains...))
get_acceptance_rates(x::Chain) = x.accept[x.theta[1].G]


## Calculate class-membership probabilities for each subject
function class_prob(xi, Xp)
    cp = exp.(Xp * xi')  ## N x G
    cp ./ sum(cp, dims=2)[:,1] ## normalize each row
end

function build_cov_matrix(S::Vector{T}, L::Matrix{T}) where T
    SD = Diagonal(S)
    Array(Symmetric(SD*L*L'*SD))
end

build_cov_matrix(theta::Tp, g::Int) where Tp <: AbstractParam = buidl_cov_matrix(theta.S[g,:], theta.L[g,:,:])

function post_cind_prob(theta::ParamSpline, data)
    @unpack Y, times, n, Xl, Z, Xp, Xe, T, E, haz_basis, cumhaz_basis = data
    @unpack xi, beta, mu, sigma, w, delta, S, L = theta
    
    G, q = size(mu)
    N = length(data.T)
    cp = class_prob(xi, Xp)
    res = Matrix{Float64}(undef, N, G)

    for g in 1:G
        Sigma = build_cov_matrix(S[g,:], L[g,:,:])
        
        for i in 1:N
            nt = 1:n[i]        
            lll_vec = loglik_p_long(beta[g,:], mu[g,:], Sigma, sigma, Y[i,1:n[i]], times[i,1:n[i]], n[i], Xl[i,1:n[i],:], Z[i,1:n[i],:])
            lle_vec = loglik_p_surv(T[i], E[i], Xe[i:i,:], haz_basis[i:i,:], cumhaz_basis[i:i,:], w[g,:], delta[g,:])
            res[i,g] = lll_vec + lle_vec + log(cp[g])
        end           
    end
    res
end

function post_cind_prob(theta::Param, data)
    @unpack Y, times, n, Xl, Z, Xp, Xe, T, E = data
    @unpack xi, beta, mu, sigma, bl_shape, bl_scale, delta, S, L = theta
    
    G, q = size(mu)
    N = length(data.T)
    cp = class_prob(xi, Xp)
    res = Matrix{Float64}(undef, N, G)

    for g in 1:G
        Sigma = build_cov_matrix(S[g,:], L[g,:,:])
        
        for i in 1:N
            nt = 1:n[i]        
            lll_vec = loglik_p_long(beta[g,:], mu[g,:], Sigma, sigma, Y[i,1:n[i]], times[i,1:n[i]], n[i], Xl[i,1:n[i],:], Z[i,1:n[i],:])
            lle_vec = loglik_p_surv(T[i], E[i], Xe[i:i,:], bl_shape[g], bl_scale[g], delta[g,:])
            res[i,g] = lll_vec + lle_vec + log(cp[g])
        end           
    end
    res
end

## sample from discrete distribution with probabilities given by exp.(alpha)/sum(exp.(alpha)) using Gumbel-max trick
sample_discrete(alpha) = last(findmax(alpha + rand(Gumbel(), length(alpha))))

function sample_cind(theta::Tp, data::Data) where Tp <: AbstractParam    
    alpha = post_cind_prob(theta, data)   
    [sample_discrete(r) for r in eachrow(alpha)]
end

function sample_sigma(b, beta, mu, sigma, Xl, Z, Y, n, times, cind, prior)
    N = size(Xl)[1]
    q = size(Xl)[3]    
    v = 0.0

    for i in 1:N
        g = cind[i]
        u = b[i,:] ##mu[g,:] + b[i,:]
        
        for j in 1:n[i]
            v += (Y[i,j] - dot(Z[i,j,:], u) - dot(Xl[i,j,:], beta[g,:]))^2
        end
    end
    v /= 2
    
    log_target(x::Float64) = logpdf(InverseGamma(sum(n)/2 - 1, v), exp(2*x)) + logpdf(prior, exp(x)) + x

    y, ac = metropolis(log_target, log(sigma), rand(Normal(log(sigma), 0.1)))

    exp(y), ac
end

function sample_mu_conj(Sigma, beta, mu, sigma, Xl, Z, Y, n, prior_mu)
    N = size(Y,1)
    q = length(mu)

    V0 = inv(cov(prior_mu))
    ll_mean_mu = zeros(Float64, q)
    ll_inv_cov_mu = zeros(Float64, q, q)
      
    for i in 1:N
        tt = 1:n[i]
        Zi = Z[i,tt,:]
        Xli = Xl[i,tt,:]
        C = inv(Array(Symmetric(Zi*Sigma*Zi' + sigma^2*diagm(0 => ones(Float64, n[i])))))
        ll_mean_mu += Zi'*C*(Y[i,tt] - Xli*beta)
        ll_inv_cov_mu += Zi'*C*Zi
    end

    post_cov = inv(Symmetric(V0 + ll_inv_cov_mu))    
    post_mean = post_cov * (V0*mean(prior_mu) + ll_mean_mu)
    rand(MvNormal(post_mean, post_cov)), 1
end

function sample_beta_conj(theta, data, prior_beta)
    @unpack beta, mu, sigma, S, L, cind = theta
    @unpack Xl, Z, Y, n = data
    
    N = size(Y,1)
    l = size(beta,2)
    G, q = size(mu)
    
    V0 = inv(cov(prior_beta))

    ll_mean_beta = zeros(Float64, l)
    ll_inv_cov_beta = zeros(Float64, l, l)

    Sigma = zeros(eltype(beta), G, q, q)
    for g in 1:G
        Sigma[g,:,:] = build_cov_matrix(S[g,:], L[g,:,:])
    end

    for i in 1:N
        tt = 1:n[i]
        Zi = Z[i,tt,:]
        Xli = Xl[i,tt,:]
        g = cind[i]
        C = inv(Array(Symmetric(Zi*Sigma[g,:,:]*Zi' + sigma^2*diagm(0 => ones(Float64, n[i])))))
        ll_mean_beta += Xli'*C*(Y[i,tt] - Zi*mu[g,:])
        ll_inv_cov_beta += Xli'*C*Xli
    end

    post_cov = inv(Symmetric(V0 + ll_inv_cov_beta))
    post_mean = post_cov * (V0*mean(prior_beta) + ll_mean_beta)
    rand(MvNormal(post_mean, post_cov)), 1
end

function sample_mu_beta_conj(Sigma, beta, mu, sigma, Xl, Z, Y, n, prior_mu, prior_beta)
    N = size(Y,1)
    q = length(mu)
    l = length(beta)    

    mb0 = vcat(mean(prior_mu), mean(prior_beta))
    V0 = zeros(Float64, q+l, q+l)
    ## assuming independent priors for mu and beta
    V0[1:q, 1:q] = inv(cov(prior_mu))
    V0[(q+1):end, (q+1):end] = inv(cov(prior_beta))

    ll_mean_mu = zeros(Float64, q)
    ll_mean_beta = zeros(Float64, l)
    ll_inv_cov_mu = zeros(Float64, q, q)
    ll_inv_cov_beta = zeros(Float64, l, l)
      
    for i in 1:N
        tt = 1:n[i]
        Zi = Z[i,tt,:]
        Xli = Xl[i,tt,:]
        C = inv(Array(Symmetric(Zi*Sigma*Zi' + sigma^2*diagm(0 => ones(Float64, n[i])))))
        ll_mean_mu += Zi'*C*(Y[i,tt] - Xli*beta)
        ll_mean_beta += Xli'*C*(Y[i,tt] - Zi*mu)
        ll_inv_cov_mu += Zi'*C*Zi
        ll_inv_cov_beta += Xli'*C*Xli
    end

    ll_inv_cov = zeros(Float64, q+l, q+l)
    ll_inv_cov[1:q, 1:q] = ll_inv_cov_mu
    ll_inv_cov[(q+1):end, (q+1):end] = ll_inv_cov_beta
    
    post_cov = inv(Symmetric(V0 + ll_inv_cov))    
    post_mean = post_cov * (V0*mb0 + vcat(ll_mean_mu, ll_mean_beta))

    mu_beta = rand(MvNormal(post_mean, post_cov))
    mu_beta[1:q], mu_beta[(q+1):end], 1
end

function inv_cov_SL(S, L)
    ##try
        invSL = inv(LowerTriangular(Diagonal(S)*L))
        return invSL'invSL
    ##catch ex
    ##    @info S, L
    ##    error("singular")
    ##end
    ##return diagm(0 => ones(Float64, length(S)))
end

function sample_b_conj(theta, data)
    @unpack G, b, beta, mu, sigma, cind, S, L = theta
    @unpack T, E, Z, Xe, Xl, Z, Y, n, times = data

    N = length(T)
    q = size(b,2)
    z = Matrix{Float64}(undef, N, q)
    invSigma = [inv_cov_SL(S[g,:], L[g,:,:]) for g in 1:G]
    z0 = rand(MvNormal(diagm(0 => ones(eltype(b), q))), N)
    
    for i in 1:N
        g = cind[i]               
        tt = 1:n[i]
        Zi = Z[i,tt,:]
        ll_mean = Zi'*(Y[i,tt] - Xl[i,tt,:]*beta[g,:])
        ll_inv_cov = Zi'*Zi / sigma^2
        
        V = Symmetric(invSigma[g] + ll_inv_cov)
        V_chol = cholesky!(V; check=false)
        
        if !issuccess(V_chol)
            z[i,:] = b[i,:]
        else
            VL = inv(V_chol.L)
            post_mean = VL'*VL*(invSigma[g]*mu[g,:] + ll_mean / sigma^2)
            z[i,:] = post_mean + VL'*z0[:,i]
        end
    end
    z, 1
end

function update_lmm_params_common!(theta::T, accept::T, data::Data, prior) where T <: AbstractParam
    G, q = size(theta.mu)

    theta.beta[1,:], ac = sample_beta_conj(theta, data, prior.beta_mv)
    
    for g in 1:G
        theta.beta[g,:] = theta.beta[1,:]
    end
    accept.beta .+= ac
    
    for g in 1:G
        sel = theta.cind .== g

        if sum(sel) > 0
            data_g = slice_data(data, sel)

            Sigma = build_cov_matrix(theta.S[g,:], theta.L[g,:,:])
            
            theta.mu[g,:], ac = sample_mu_conj(Sigma, theta.beta[g,:], theta.mu[g,:], theta.sigma,
                                               data_g.Xl, data_g.Z, data_g.Y, data_g.n, prior.mu_mv)
            accept.mu[g,:] .+= ac
        else
            theta.mu[g,:] = rand(prior.mu_mv)
            accept.mu[g,:] .+= 1
        end
    end
    
    theta.b, ac = sample_b_conj(theta, data)
    accept.b .+= ac

    theta.sigma, ac = sample_sigma(theta.b, theta.beta, theta.mu, theta.sigma, data.Xl, data.Z, data.Y, data.n, data.times, theta.cind, prior.sigma)
    accept.sigma += ac    
end

function update_lmm_params_mix!(theta::T, accept::T, data::Data, prior) where T <: AbstractParam
    G, q = size(theta.mu)
    
    for g in 1:G
        sel = theta.cind .== g

        if sum(sel) > 0
            data_g = slice_data(data, sel)

            Sigma = build_cov_matrix(theta.S[g,:], theta.L[g,:,:])
            
            theta.mu[g,:], theta.beta[g,:], ac = sample_mu_beta_conj(Sigma, theta.beta[g,:], theta.mu[g,:], theta.sigma, data_g.Xl, data_g.Z, data_g.Y, data_g.n, prior.mu_mv, prior.beta_mv)
            accept.beta[g,:] .+= ac
            accept.mu[g,:] .+= ac
        else
            theta.mu[g,:] = rand(prior.mu_mv)
            accept.mu[g,:] .+= 1

            theta.beta[g,:] = rand(prior.beta_mv)
            accept.beta[g,:] .+= 1
        end
    end
    
    theta.b, ac = sample_b_conj(theta, data)
    accept.b .+= ac

    theta.sigma, ac = sample_sigma(theta.b, theta.beta, theta.mu, theta.sigma, data.Xl, data.Z, data.Y, data.n, data.times, theta.cind, prior.sigma)
    accept.sigma += ac    
end

function transform_zero_sum(xi)
    G = size(xi,1)
    A = diagm(0 => ones(eltype(xi), G)) .- 1/G
    A*xi
end

function transform_ref_group(xi)
    G, p = size(xi)
    A = diagm(0 => ones(eltype(xi), G-1)) .- 1/G    
    inv(A)*xi[1:(G-1),:]
end

function sample_xi_NUTS(sampler, xi, cind, Xp)
    G, p = size(xi)
    ##init = xi[1:(G-1),:]
    init = transform_ref_group(xi)
    res, ac = nuts_mlogit_step!(sampler, cind_to_matrix(cind, G), init)
    ##vcat(res.xi, zeros(eltype(res.xi), 1, p)), ac
    new_xi = transform_zero_sum(vcat(res.xi, zeros(eltype(res.xi), 1, p)))
    new_xi, ac
end
 
function update_cp_params!(theta::T, accept::T, data::Data, prior, sampler) where T <: AbstractParam
    G, q = size(theta.mu)

    if G == 1
        theta.cind .= 1
        return loglik(theta, data)
    else
        theta.cind = sample_cind(theta, data)
        accept.cind .+= 1
        
        theta.xi, ac = sample_xi_NUTS(sampler, theta.xi, theta.cind, data.Xp)
        ##theta.xi, ac = sample_xi(theta.xi, theta.cind, data.Xp)
        accept.xi .+= ac       
    end    
end

function cind_to_matrix(cind::Vector{Int}, G::Int)
    Y = zeros(Int, length(cind), G)
    for i in 1:length(cind)
        Y[i, cind[i]] = 1
    end
    Y
end

function init_NUTS_samplers(warmup, G, theta, data, prior, cp=Dict{Int, NUTSGibbsSampler}(),
                            surv=Dict{Int, NUTSGibbsSampler}(),
                            cov=Dict{Int, NUTSGibbsSampler}(); common_delta=false)
    if !haskey(surv, G)
        if G > 1
            cp[G] = nuts_mlogit_init(cind_to_matrix(theta.cind, G), data.Xp, prior.xi, warmup)
            @debug "Initialized NUTS sampler for class membership regression (G = $(G))"
        end
        
        if common_delta
            surv[G] = surv_model_init2(theta, data, prior, warmup)
        else
            surv[G] = surv_model_init(theta, data, prior, warmup)
        end
        @debug "Initialized NUTS sampler for survival submodel (G = $(G))"
        
        ##@info "Initializing NUTS sampler for LMM submodel"
        ##lmm[G] = NUTS_JLCM_lmm_init(theta, data, prior)

        cov[G] = cov_model_init(theta, data, prior, warmup)
        @debug "Initialized NUTS sampler for covariance model (G = $(G))"
        
        ##@info "Initializing NUTS sampler for JLCM model (G = $(G))"
        ##cov[G] = NUTS_JLCM_init(theta, data, prior)
    end
    
    cp, surv, cov
end

function sampling(M::Int, data::Data, par_type::Type{Tp}, prior::Tprior=Prior(data), prop::Tprior=prior;
                  chains=1, G=1, warmup=Int(floor(M/2)), common_delta=false, common_beta=false,
                  return_warmup=false) where {Tp <: AbstractParam, Tprior <: AbstractPrior}
    control = Control(M, warmup, chains, return_warmup, common_delta, common_beta, 0.8)
    sampling(control, data, par_type, prior, prop; G=G)
end

function sampling(control::Control, data::Data, par_type::Type{Tp}, prior::Tprior=Prior(data), prop::Tprior=prior; G=1) where {Tp <: AbstractParam, Tprior <: AbstractPrior}

    sampling(control, data, [Tp(data,G) for k in 1:control.chains], prior, prop)
end

function sampling(M, data::Data, x::T; warmup=floor(Int, M/2), from_init=true) where T <: AbstractFit
    c = x.control
    ctrl = Control(M, warmup, c.chains, c.save_warmup, c.common_delta, c.common_beta, c.adapt_delta)

    if from_init
        init = [xx.init for xx in x.chains]
    else
        init = [xx.theta[end] for xx in x.chains]
    end
    
    sampling(ctrl, data, init, x.chains[1].prior)
end

function sampling(control::Control, data::Data, init::Vector{Tp}, prior::Tprior=Prior(data), prop::Tprior=prior) where {Tp <: AbstractParam, Tprior <: AbstractPrior}
    
    @assert length(init) == control.chains
    
    chains = @showprogress pmap(1:control.chains) do k
        @info "chain $(k) of $(control.chains)"
        sampling(control, data, init[k]; prior=prior, prop=prop)
    end

    Fit(control, chains)    
end

function sampling(control::Control, data::Data, init::Tp=Param(data, 1);
                  prior::Tpr=Prior(), prop::Tpr=prior) where {Tp <: AbstractParam, Tpr <: AbstractPrior}
    G = init.G
    M = control.iter
    warmup = control.warmup
    
    offset = control.save_warmup ? 0 : warmup
        
    @info "Taking $(M) samples including $(warmup) warmup iterations in $(G)-class model"
    
    loglik_ar = Matrix{Float64}(undef, M-offset, length(data.T))    
    theta_stor = Vector{Tp}(undef, M-offset)
    
    theta = deepcopy(init)

    if offset == 0
        theta_stor[1] = deepcopy(theta)
        loglik_ar[1,:] = loglik_p(theta, data)
    end
    
    accept = Tp(data, G; set_zero=true)

    cp_sampler, surv_sampler, cov_sampler = init_NUTS_samplers(warmup, G, theta, data, prior; common_delta=control.common_delta)
    ac = 0.0
    
    ## @showprogress 1 "Sampling..."
    for t in 2:M
        if G > 1
            update_cp_params!(theta, accept[G], data, prior, cp_sampler[G])
        end

        if control.common_beta
            update_lmm_params_common!(theta, accept, data, prior)
        else
            update_lmm_params_mix!(theta, accept, data, prior)
        end
        
        theta, ac = cov_model_step!(cov_sampler[G], theta)
        accept.S .+= ac
        accept.L .+= ac
        accept.sigma += ac

        if control.common_delta
            theta, ac = surv_model_step2!(surv_sampler[G], theta)
        else
            theta, ac = surv_model_step!(surv_sampler[G], theta)
        end
        
        accept.delta .+= ac
        accept.bl_shape .+= ac
        accept.bl_scale .+= ac        

        theta.G0 = sum([sum(theta.cind .== g) == 0 for g in 1:G])
                
        if t > offset
            ll = loglik_p(theta, data)
            loglik_ar[t-offset,:] = ll
            theta_stor[t-offset] = deepcopy(theta)
        end
    end
    
    Chain{Tp,Tpr}(theta_stor, accept, loglik_ar, prior, init)
end
