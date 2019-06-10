## sample random effects given latent class membership, observed data and parameters
function sample_rand(g::Int, data::Data, param::Param)
    @unpack beta, mu, S, L, sigma, b = param
    @unpack Y, times, n, Xl, Z = data

    N, q = size(b)
    z = Matrix{Float64}(undef, N, q)
    invSigma = inv_cov_SL(S[g,:], L[g,:,:])
    
    tt = 1:n[1]
    Zi = Z[1,tt,:]    
    ll_mean = Zi'*(Y[1,tt] - Xl[1,tt,:]*beta[g,:])
    ll_inv_cov = Zi'*Zi / sigma^2    
    post_cov = inv(Symmetric(invSigma + ll_inv_cov))
    post_mean = post_cov*(invSigma*mu[g,:] + ll_mean / sigma^2)
    
    rand(MvNormal(post_mean, post_cov))       
end

f1(t, eta=-1.5) = (1 .+ t).^eta .- 1
f2(t, nu=0.0) = t.^(nu+1) ./ (t.+1).^nu

## sample longitudinal trjactory given latent class membership, random effects, covariates and parameters
function sample_lmm(time::AbstractVector{T}, g::Int, b::Vector{T}, data::Data, param::Param) where T <: Real
    @unpack beta, sigma = param
    @unpack Xl, n = data

    X = Xl[1,1,:]
    y = zeros(T, length(time))
    
    for j in 1:length(time)        
        Z = [1, f1(time[j]), f2(time[j])]
        Xl = vcat(X[1:3], X[4:6]*f1(time[j]), X[7:9]*f2(time[j]))
        mean_y = dot(Xl, beta[g,:]) + dot(Z, b)
        y[j] = mean_y ##rand(Normal(mean_y, sigma))
    end
    return y
end

surv_g(time, shape, scale, delta, Xe) = exp.(-exp(dot(Xe, delta))*weibull_cumhaz(time, shape, scale))

## prediction of conditional survival function and longitudinal trajectory conditional on new_data
## new_data is interpreted as data from a new subject if out_of_sample=true or as additional data from an existing
## subject in data if out_of_sample=false (for within sample prediction new data can be same as old data)
## for new subjects latent class indicator and random effects are sampled
## for existing subjects the existing posterior samples of latent class indicator and random effects are used    
function predict(time::AbstractVector{T}, new_data::Data, x::Tf; out_of_sample=true) where {T <: Real, Tf <: Fit}
    param = get_samples(x)
    
    if length(new_data.T) > 1
        @warn "Only using first subject in new_data"
    end

    t0 = new_data.T[1]
    @assert minimum(time) >= t0

    obs_t = new_data.times[1, 1:new_data.n[1]]
    obs_y = new_data.Y[1,1:new_data.n[1]]

    if t0 > maximum(obs_t)
        time = vcat(t0, sort(time))
    end
    
    M = length(param)
    b = Matrix{T}(undef, M, size(param[1].b,2))
    y = Matrix{T}(undef, M, length(t))    
    S = Matrix{T}(undef, M, length(t))
    g = Vector{Int}(undef, M)
        
    for m in 1:M
        theta = param[m]

        if out_of_sample
            g[m] = sample_cind(theta, new_data)[1] ##sample_class(new_data, theta)
            b[m,:] = sample_rand(g[m], new_data, theta)
        else
            g[m] = theta.cind[1]
            b[m,:] = theta.b[1,:]
        end
        
        y[m,:] = sample_lmm(time, g[m], b[m,:], new_data, theta)
        S0 = surv_g(new_data.T[1], theta.bl_shape[g[m]], theta.bl_scale[g[m]], theta.delta[g[m],:], new_data.Xe[1,:])        
        S[m,:] = min.(surv_g(time, theta.bl_shape[g[m]], theta.bl_scale[g[m]], theta.delta[g[m],:], new_data.Xe[1,:]) ./ S0, 1.0)
    end
       
    summary_predict(obs_t, obs_y, time, y, S)
end

function summary_predict(obs_t, obs_y, t, y, S)
    S0 = repeat([1.0], inner=length(obs_t))
    
    S_ci_lo = vcat(S0, [quantile(S[:,i], 0.025) for i in 1:length(t)])
    S_ci_hi = vcat(S0, [quantile(S[:,i], 0.975) for i in 1:length(t)])

    y_ci_lo = vcat(obs_y, [quantile(y[:,i], 0.025) for i in 1:length(t)])
    y_ci_hi = vcat(obs_y, [quantile(y[:,i], 0.975) for i in 1:length(t)])
    
    DataFrame(time=vcat(obs_t, t), type=vcat(repeat(["observed"], inner=length(obs_t)), repeat(["predicted"], inner=length(t))),
              y=vcat(obs_y, mean(y, dims=1)[1,:]), sd_y=vcat(zeros(Float64, length(obs_t)), sqrt.(var(y, dims=1))[1,:]), ci_lo_y=y_ci_lo, ci_hi_y=y_ci_hi,
              S=vcat(S0, mean(S, dims=1)[1,:]), sd_S=vcat(zeros(Float64, length(obs_t)), sqrt.(var(S, dims=1)[1,:])), ci_lo_S=S_ci_lo, ci_hi_S=S_ci_hi)
end

function plot_predict(df::DataFrame; use_R=true)
    if use_R
        R"library(ggplot2)"
        @rput df
        R"ggplot(df, aes(x=df$time, y=df$y, ymin=df$ci_lo_y, ymax=df$ci_hi_y)) + geom_line() + geom_ribbon()"
        R"ggplot(df, aes(x=df$time, y=df$S, ymin=df$ci_lo_S, ymax=df$ci_hi_S)) + geom_line() + geom_ribbon()"
    else
        df_obs = df[df.type .== "observed",:]
        df_pred = df[df.type .== "predicted",:]
        
        p1 = plot(layer(x=df_pred.time, y=df_pred.y, ymin=df_pred.ci_lo_y, ymax=df_pred.ci_hi_y, Geom.line, Geom.ribbon),
                  layer(x=df_obs.time, y=df_obs.y, Geom.point), Guide.xlabel("Time"), Guide.ylabel("log(PSA)"),
                  Coord.cartesian(xmin=0, xmax=15, ymin=0, ymax=10))
        p2 = plot(x=df_pred.time, y=df_pred.S, ymin=df_pred.ci_lo_S, ymax=df_pred.ci_hi_S, Geom.line, Geom.ribbon, Guide.xlabel("Time"), Guide.ylabel("Survival Probability"), Coord.cartesian(xmin=0, xmax=15, ymin=0, ymax=1))

        hstack(p1, p2)
    end
end
