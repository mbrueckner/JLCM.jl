mutable struct CovProblem{T <: Real, S <: Int} <: NUTSGibbs.AbstractProblem
    "Observations."
    X::Matrix{T}
    "Degrees of freedom for half-T prior for standard deviations"
    v::T
    "Degrees of freedom for LKJ cholesky distribution"
    eta::T
    "Number of classes"
    G::S
    "Class indicators"
    cind::Vector{S}
end

function nuts_cov_transformation(p::CovProblem)
    d = size(p.X, 2)
    as((S = as(Matrix, asℝ₊, p.G, d), y = as(Matrix, p.G, Int(d*(d-1)/2))))
end


is_corr_cholfac(L) = isapprox(sum(L.^2, dims=2), ones(eltype(L), size(L,1))) && all(diag(L) .> 0.0)
invquad(L, x) = dot(x, L' \ (L \ x))
mvn_logpdf(x, mu, L) = -length(mu)/2 * log(2*pi) - log(det(L)) - invquad(L, x-mu)/2

function log_target(S, y, X, v, eta)
    n, d = size(X)
    L = transform_free_to_chol(y, d)
    distr_L = LKJcorrChol(d, eta)
    SL = Diagonal(S)*L
    
    (is_corr_cholfac(L) && all(isfinite.(S)) && all(S .> 0.00001)) || return -Inf

    res = -n*log(prod(diag(SL)))
    for i in 1:n
        res -= invquad(SL, X[i,:])/2
    end

    res += sum(logpdf.(Cauchy(0, 2.5), S)) + logpdf(distr_L, L) + log_jacobian_det_free_to_chol(y, d, L)
    isfinite(res) || return -Inf
    res
end

function (problem::CovProblem)(theta)
    @unpack cind, X, v, eta, G = problem
    @unpack S, y = theta

    d = size(X,2)

    res = 0.0
    for g in 1:G
        res += log_target(S[g,:], y[g,:], X[cind .== g,:], v, eta)
    end
    
    isfinite(res) || return -Inf
    res
end

function cov_model_init(theta, data, prior, warmup)
    G, q = size(theta.S)
    X = theta.b .- theta.mu[theta.cind,:]    
    problem = CovProblem(X, 1.0, 1.0, G, theta.cind)
    NUTSGibbs.init(problem, nuts_cov_transformation, warmup)
end

function cov_model_step!(x::NUTSGibbsSampler, theta::Tp) where Tp <: AbstractParam
    G, q = size(theta.S)
    x.problem.X = theta.b .- theta.mu[theta.cind,:]
    x.problem.cind = theta.cind
    
    y = Matrix{eltype(theta.L)}(undef, G, Int(q*(q-1)/2))
    for g in 1:G
        y[g,:] = transform_chol_to_free(LowerTriangular(theta.L[g,:,:]))
    end

    res, ac = NUTSGibbs.step(x, (S=theta.S, y=y))

    theta.S = res.S

    for g in 1:G
        theta.L[g,:,:] = transform_free_to_chol(res.y[g,:], q)
    end
    
    return theta, ac
end
