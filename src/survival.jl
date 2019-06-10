## Calculate survival function for all subjects on a grid given (shape,scale,delta) given class-membership
## size(Xe) = N*r
function surv_weibull(grid::Vector{T}, shape::T, scale::T, delta::Vector{T}, Xe::Vector{T}) where T <: Real
    exp.(-exp(dot(Xe,delta)) * weibull_cumhaz(grid, shape, scale))
end

function surv(time::Vector{T}, param::Param, data::Data) where T <: Real
    @unpack bl_shape, bl_scale, delta, cind = param

    N = size(data.Xe, 1)
    res = zeros(T, N, length(time))
    
    for i in 1:N
        g = cind[i]
        res[i,:] = surv_weibull(time, bl_shape[g], bl_scale[g], delta[g,:], data.Xe[i,:])
    end
    res
end

function surv_mixture(time::Vector{T}, param::Param, data::Data) where T <: Real
    @unpack G, xi, bl_shape, bl_scale, delta = param

    N = size(data.Xe, 1)
    res = zeros(T, N, length(time))

    cp = class_prob(xi, data.Xp)

    for i in 1:N
        for g in 1:G
            res[i,:] += cp[i,g].*surv_weibull(time, bl_shape[g], bl_scale[g], delta[g,:], data.Xe[i,:])
        end
    end
    res
end

function surv_spline(grid::Vector{T}, w::Vector{T}, delta::Vector{T}, Xe::Vector{T}, cumhaz_basis::Matrix{T}) where T <: Real
    exp.(-exp(dot(Xe,delta)) .* cumhaz_basis*exp.(w))
end

function surv(time::Vector{T}, param::ParamSpline, data::Data) where T <: Real
    @unpack w, delta, cind = param

    haz_basis, cumhaz_basis = eval_spline(data, time)
    
    N = size(data.Xe, 1)
    res = zeros(T, N, length(time))
    
    for i in 1:N
        g = cind[i]
        res[i,:] = surv_spline(time, w[g,:], delta[g,:], data.Xe[i,:], cumhaz_basis)
    end
    res
end

function surv_mixture(time::Vector{T}, param::ParamSpline, data::Data) where T <: Real
    @unpack G, xi, w, delta = param

    haz_basis, cumhaz_basis = eval_spline(data, time)
    
    N = size(data.Xe, 1)
    res = zeros(T, N, length(time))

    cp = class_prob(xi, data.Xp)

    for i in 1:N
        for g in 1:G
            res[i,:] += cp[i,g].*surv_spline(time, w[g,:], delta[g,:], data.Xe[i,:], cumhaz_basis)
        end
    end
    res
end

function km_survival(data::Data, grid=sort(unique(data.T))) where T <: AbstractParam
    R"library(survival)"
    df = DataFrame(time=data.T, status=data.E)
    @rput df
    @rput grid
    R"x <- survfit(Surv(time, status) ~ 1, data=df)"
    R"fs <- approxfun(x$time, x$surv, f=1, rule=2, method='constant')"
    R"fe <- approxfun(x$time, x$std.err, f=1, rule=2, method='constant')"
    R"surv <- fs(grid)"
    R"se <- fe(grid)"
    R"res <- data.frame(time=grid, surv=surv, ci_lo=pmax(0, surv - 1.96*se), ci_hi=pmin(1, surv + 1.96*se))"
    @rget res
    res
end

function marginal_survival(x::T, data::Data; grid=sort(unique(data.T))) where T <: Fit
    marginal_survival(merge(x.chains...), data; grid=grid)
end

function marginal_survival(x::T, data::Data; start=1.0, grid=sort(unique(data.T))) where T <: Chain
    ##df = by(conditional_survival(x, data, grid), :time) do z
    ##    (surv=mean(z.mean), ci_lo=mean(z.ci_lo), ci_hi=mean(z.ci_hi))
    ##end

    @assert minimum(grid) >= start
    
    M = length(x)
    mn = Vector{Float64}(undef, length(grid))
    ci_lo = Vector{Float64}(undef, length(grid))
    ci_hi = Vector{Float64}(undef, length(grid))
    S = Matrix{Float64}(undef, M, length(grid))

    z = get_samples(x)
    
    for t in 1:M
        S[t,:] = mean(surv_mixture(grid .- start, z[t], data), dims=1)[1,:]
    end
    
    for j in 1:length(grid)
        Sv = view(S, :, j)
        mn[j] = mean(Sv)
        ci_lo[j] = quantile(Sv, 0.025)
        ci_hi[j] = quantile(Sv, 0.975)
    end
    
    df = DataFrame(time=grid, surv=mn, ci_lo=ci_lo, ci_hi=ci_hi)
    
    km = km_survival(data, grid)
    df = vcat(df, km)
    df.method = repeat(["mc", "km"], inner=length(unique(grid)))
    df
end

function class_marginal_survival(x::Fit{T}, data::Data; start=1.0, grid=sort(unique(data.T))) where T <: AbstractParam
    ##df = by(conditional_survival(x, data, grid), :time) do z
    ##    (surv=mean(z.mean), ci_lo=mean(z.ci_lo), ci_hi=mean(z.ci_hi))
    ##end

    @assert minimum(grid) >= start

    G = x.theta[1].G
    
    M = length(x.theta)
    mn = Matrix{Float64}(undef, G, length(grid))
    ci_lo = Matrix{Float64}(undef, G, length(grid))
    ci_hi = Matrix{Float64}(undef, G, length(grid))
    S = ones(Float64, M, G, length(grid))
    
    for t in 1:M
        par = x.theta[t]
        for g in 1:G
            sel = par.cind .== g
            if sum(sel) > 0
                S[t,g,:] = mean(surv(grid .- start, par, slice_data(data, sel)), dims=1)[1,:]
            end
        end
    end
    
    for j in 1:length(grid)
        for g in 1:G
            Sv = view(S, :, g, j)
            mn[g,j] = mean(Sv)
            ci_lo[g,j] = quantile(Sv, 0.025)
            ci_hi[g,j] = quantile(Sv, 0.975)
        end
    end
    
    df = DataFrame(time=repeat(grid, inner=G), class=repeat(1:G, outer=length(grid)), surv=vec(mn), ci_lo=vec(ci_lo), ci_hi=vec(ci_hi))
    
    ##km = km_survival(data, grid)
    ##df = vcat(df, km)
    df.method = df.class ##repeat(["mc", "km"], inner=length(unique(grid)))
    df
end

function conditional_survival(x::Tf, Xe::Vector{Tx}, Xp::Vector{Tx}, data::Data; grid=sort(unique(data.T)), start=1.0) where {Tf <: Fit, Tx <: Real}
    conditional_survial(merge(x.chains...), Xe, Xp, data; grid=grid, start=start)
end

function conditional_survival(x::Tf, Xe::Vector{Tx}, Xp::Vector{Tx}, data::Data; grid=sort(unique(data.T)), start=1.0) where {Tf <: Chain, Tx <: Real}
    M = length(x.theta)

    @assert minimum(grid) >= start
                        
    ##mn = SharedArray{Float64,1}((length(grid)))
    ##ci_lo = SharedArray{Float64,1}((length(grid)))
    ##ci_hi = SharedArray{Float64,1}((length(grid)))

    mn = Vector{Float64}(undef, length(grid))
    ci_lo = Vector{Float64}(undef, length(grid))
    ci_hi = Vector{Float64}(undef, length(grid))
    S = Matrix{Float64}(undef, M, length(grid))

    z = get_samples(x)
    
    for t in 1:M
        S[t,:] = surv_mixture(grid .- start, z[t], Xe, Xp, data)
    end
    
    for j in 1:length(grid)
        Sv = view(S, :, j)
        mn[j] = mean(Sv)
        ci_lo[j] = quantile(Sv, 0.025)
        ci_hi[j] = quantile(Sv, 0.975)
    end
    
    DataFrame(time=grid, mean=mn, ci_lo=ci_lo, ci_hi=ci_hi)
end
