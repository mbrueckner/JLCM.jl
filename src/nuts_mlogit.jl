
mutable struct MultiLogitProblem{Ty, TX, Tp} <: NUTSGibbs.AbstractProblem
    y::Ty
    X::TX
    prior::Tp
end

function (problem::MultiLogitProblem)(theta)
    @unpack y, X, prior = problem
    @unpack xi = theta

    N, p = size(X)
    G = size(xi, 1)
    ll = 0.0

    for i in 1:N
        tmp = zeros(eltype(xi), G+1)
        ##tmp = 1.0
        
        for g in 1:G
            xx = dot(X[i,:], xi[g,:])
            tmp[g] = xx
            ##tmp += exp(xx)
            ll += y[i,g]*xx
        end

        ll -= logsumexp(tmp)
        ##ll -= log(tmp)
    end

    for g in 1:G
        ll += sum(logpdf.(prior, xi[g,:]))
    end

    isfinite(ll) || return -Inf
    
    ll
end
                    
function mlogit_transformation(p::MultiLogitProblem)
    D = size(p.X,2)
    G = size(p.y,2)
    as((xi = as(Array, G-1, D),))
end

function nuts_mlogit_init(y, X, prior, warmup)
    problem = MultiLogitProblem(y, X, prior)
    NUTSGibbs.init(problem, mlogit_transformation, warmup)
end

function nuts_mlogit_step!(x::NUTSGibbsSampler, y, xi)   
    x.problem.y = y    
    return NUTSGibbs.step(x, (xi=xi,))
end
