mutable struct Param <: AbstractParam
    ## number of latent classes
    G::Int

    ## number of empty latent classes
    G0::Int
    
    ## multinomial model
    xi::Matrix{Float64} # G x (p+1) matrix

    ### longitudinal model
    ## fixed effects
    beta::Matrix{Float64} # G x nfix

    # ## random effects
    mu::Matrix{Float64} # G x nrand

    ## random effects standard errors
    S::Matrix{Float64} ##Vector{Float64} ## q
    
    # ## random effects correlation matrix Cholesky factor
    L::Array{Float64,3} ##Matrix{Float64} # nrand x nrand
    
    # ## measurement error standard deviation
    sigma::Float64

    ### survival model
    bl_shape::Vector{Float64} # G
    bl_scale::Vector{Float64} # G
    delta::Matrix{Float64} # G x r

    b::Matrix{Float64}
    cind::Vector{Int}
end

mutable struct ParamSpline <: AbstractParam
    ## number of latent classes
    G::Int

    ## number of empty latent classes
    G0::Int
    
    ## multinomial model
    xi::Matrix{Float64} # G x (p+1) matrix

    ### longitudinal model
    ## fixed effects
    beta::Matrix{Float64} # G x nfix

    # ## random effects
    mu::Matrix{Float64} # G x nrand

    ## random effects standard errors
    S::Matrix{Float64} ##Vector{Float64} ## q
    
    # ## random effects correlation matrix Cholesky factor
    L::Array{Float64,3} ##Matrix{Float64} # nrand x nrand
    
    # ## measurement error standard deviation
    sigma::Float64

    ### survival model
    w::Matrix{Float64} # G x (knots + 4)
    delta::Matrix{Float64} # G x r

    b::Matrix{Float64}
    cind::Vector{Int}
end

Param(N, xi, beta, mu, S, L, sigma, shape, scale, delta) = Param(size(beta,1), 0, xi, beta, mu, S, L, sigma,
                                                                 shape, scale, delta, zeros(eltype(mu), N, size(mu,2)),
                                                                 sample(1:size(mu,1), N))

ParamSpline(N, xi, beta, mu, S, L, sigma, w, delta) = Param(size(beta)[1], 0, xi, beta, mu, S, L, sigma,
                                                            w, delta, zeros(eltype(mu), N, size(mu,2)),
                                                            sample(1:size(mu,1), N))

function Param(data::Data, G::Int; set_zero=false)
    N, p = size(data.Xp)
    
    ## number of hazard predictors
    r = size(data.Xe,2)

    ## number of fixed effects in LMM model
    l = size(data.Xl,3)

    ## number of random effects in LMM model
    q = size(data.Z,3)

    if set_zero
        Param(G, 0, zeros(Float64, G, p), zeros(Float64, G, l), zeros(Float64, G, q), zeros(Float64, G, q),
              zeros(Float64, G, q, q), ##diagm(0 => ones(Float64, q)),
              0.0, zeros(Float64, G), zeros(Float64, G), zeros(Float64, G, r),
              zeros(Float64, N, q), zeros(Float64, N))
    else
        param = Param(G, 0, zeros(Float64,G,p), randn(G,l), randn(G,q), ones(Float64, G, q),
                      zeros(Float64, G, q, q), ##diagm(0 => ones(Float64, q)),
                      1.0, rand(Uniform(0.5, 1.5), G), rand(G) .+ 0.05, rand(G, r) .- 0.5,
                      randn(N, q), sample(1:G, N))

        for g in 1:G
            param.L[g,:,:] = diagm(0 => ones(Float64, q))
        end
        param
    end
end

function ParamSpline(data::Data, G::Int; set_zero=false)
    N, p = size(data.Xp)
    
    ## number of hazard predictors
    r = size(data.Xe,2)

    ## number of fixed effects in LMM model
    l = size(data.Xl,3)

    ## number of random effects in LMM model
    q = size(data.Z,3)

    k = size(data.haz_basis,2)
    
    if set_zero
        ParamSpline(G, 0, zeros(Float64, G, p), zeros(Float64, G, l), zeros(Float64, G, q), zeros(Float64, G, q),
              zeros(Float64, G, q, q), ##diagm(0 => ones(Float64, q)),
              0.0, zeros(Float64, G, k), zeros(Float64, G, r),
              zeros(Float64, N, q), zeros(Float64, N))
    else
        param = ParamSpline(G, 0, zeros(Float64, G, p), randn(G, l), randn(G, q), ones(Float64, G, q),
                            zeros(Float64, G, q, q), ##diagm(0 => ones(Float64, q)),
                            1.0, rand(G, k) .- 1.0, rand(G, r) .- 0.5,
                            randn(N, q), sample(1:G, N))
        for g in 1:G
            param.L[g,:,:] = diagm(0 => ones(Float64, q))
        end
        param
    end
end

function slice_param(par::Tpar, sel) where Tpar <: AbstractParam
    new_par = deepcopy(par)
    new_par.b = par.b[sel,:]
    new_par.cind = par.cind[sel]
    new_par
end
