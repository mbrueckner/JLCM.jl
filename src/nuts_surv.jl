mutable struct SurvProblem{Tpar <: AbstractParam, Tdata <: AbstractData, Tprior <: AbstractPrior} <: NUTSGibbs.AbstractProblem
    theta::Tpar
    data::Tdata
    prior::Tprior
end

mutable struct SurvProblem2{Tpar <: AbstractParam, Tdata <: AbstractData, Tprior <: AbstractPrior} <: NUTSGibbs.AbstractProblem
    theta::Tpar
    data::Tdata
    prior::Tprior
end

## class-specific delta
function (problem::SurvProblem{Param, Tdata, Prior})(x) where Tdata <: AbstractData
    @unpack theta, data, prior = problem
    @unpack delta, bl_shape, bl_scale = x

    lp = 0.0
    for g in 1:size(delta,1)
        sel = theta.cind .== g
        lp += loglik_p_surv(data.T[sel], data.E[sel], data.Xe[sel,:], bl_shape[g], bl_scale[g], delta[g,:])
        lp += logpdf(prior.bl_shape, bl_shape[g]) + logpdf(prior.bl_scale, bl_scale[g]) + logpdf(prior.delta_mv, delta[g,:])
    end

    isfinite(lp) || return -Inf
    lp
end

function (problem::SurvProblem{ParamSpline, Tdata, PriorSpline})(x) where Tdata <: AbstractData
    @unpack theta, data, prior = problem
    @unpack delta, w = x
    lp = 0.0
    for g in 1:size(delta,1)
        sel = theta.cind .== g
        lp += loglik_p_surv(data.T[sel], data.E[sel], data.Xe[sel,:], data.haz_basis[sel,:], data.cumhaz_basis[sel,:], w[g,:], delta[g,:])
        lp += sum(logpdf.(prior.w, w[g,:])) + logpdf(prior.delta_mv, delta[g,:])
    end
    isfinite(lp) || return -Inf
    lp
end

## common delta across classes
function (problem::SurvProblem2{Param, Tdata, Prior})(x) where Tdata <: AbstractData
    @unpack theta, data, prior = problem
    @unpack delta, bl_shape, bl_scale = x

    lp = 0.0
    for g in 1:length(bl_scale)
        sel = theta.cind .== g
        lp += loglik_p_surv(data.T[sel], data.E[sel], data.Xe[sel,:], bl_shape[g], bl_scale[g], delta)
        lp += logpdf(prior.bl_shape, bl_shape[g]) + logpdf(prior.bl_scale, bl_scale[g])
    end
    lp += logpdf(prior.delta_mv, delta)
    
    isfinite(lp) || return -Inf
    lp
end

function (problem::SurvProblem2{ParamSpline, Tdata, PriorSpline})(x) where Tdata <: AbstractData
    @unpack theta, data, prior = problem
    @unpack delta, w = x

    v = var(prior.w)
    
    G = size(w,1)
    lp = 0.0
    for g in 1:G
        sel = theta.cind .== g
        lp += loglik_p_surv(data.T[sel], data.E[sel], data.Xe[sel,:], data.haz_basis[sel,:], data.cumhaz_basis[sel,:], w[g,:], delta)
        lp -= sum(w[g,:].^2)/(2*v)        
        ##lp += sum(logpdf.(prior.w, w[g,:]))
    end
    lp += logpdf(prior.delta_mv, delta)
    
    isfinite(lp) || return -Inf
    lp
end

function transformation(p::SurvProblem{Param, Tdata, Prior}) where Tdata <: AbstractData
    G, r = size(p.theta.delta)
    as((delta = as(Matrix, G, r), bl_shape = as(Vector, asℝ₊, G), bl_scale = as(Vector, asℝ₊, G)))
end

function transformation(p::SurvProblem{ParamSpline, Tdata, PriorSpline}) where Tdata <: AbstractData
    G, r = size(p.theta.delta)
    k = size(p.theta.w,2)
    as((delta = as(Matrix, G, r), w = as(Matrix, G, k)))
end

function transformation(p::SurvProblem2{Param, Tdata, Prior}) where Tdata <: AbstractData
    G, r = size(p.theta.delta)
    as((delta = as(Vector, r), bl_shape = as(Vector, asℝ₊, G), bl_scale = as(Vector, asℝ₊, G)))
end

function transformation(p::SurvProblem2{ParamSpline, Tdata, PriorSpline}) where Tdata <: AbstractData
    G, r = size(p.theta.delta)
    k = size(p.theta.w,2)
    as((delta = as(Vector, r), w = as(Matrix, G, k)))
end

function surv_model_init(theta, data, prior, warmup)
    problem = SurvProblem(theta, data, prior)    
    NUTSGibbs.init(problem, transformation, warmup)
end

function surv_model_init2(theta, data, prior, warmup)
    problem = SurvProblem2(theta, data, prior)
    NUTSGibbs.init(problem, transformation, warmup)
end

function surv_model_step!(x::NUTSGibbsSampler, theta::Param)
    x.problem.theta = theta
    res, ac = NUTSGibbs.step(x, (delta=theta.delta, bl_shape=theta.bl_shape, bl_scale=theta.bl_scale))
    theta.delta = res.delta
    theta.bl_shape = res.bl_shape
    theta.bl_scale = res.bl_scale
    theta, ac
end

function surv_model_step!(x::NUTSGibbsSampler, theta::ParamSpline)
    x.problem.theta = theta
    res, ac = NUTSGibbs.step(x, (delta=theta.delta, w=theta.w))    
    theta.delta = res.delta
    theta.w = res.w
    theta, ac
end

function surv_model_step2!(x::NUTSGibbsSampler, theta::Param)
    x.problem.theta = theta
    res, ac = NUTSGibbs.step(x, (delta=theta.delta[1,:], bl_shape=theta.bl_shape, bl_scale=theta.bl_scale))

    for g in 1:size(theta.delta,1)
        theta.delta[g,:] = res.delta
    end
    
    theta.bl_shape = res.bl_shape
    theta.bl_scale = res.bl_scale
    theta, ac    
end

function surv_model_step2!(x::NUTSGibbsSampler, theta::ParamSpline)
    x.problem.theta = theta
    res, ac = NUTSGibbs.step(x, (delta=theta.delta[1,:], w=theta.w))

    for g in 1:size(theta.delta,1)
        theta.delta[g,:] = res.delta
    end
    
    theta.w = res.w
    theta, ac    
end
