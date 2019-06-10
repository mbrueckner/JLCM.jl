inverse_perm(p) = [(1:length(p))[p .== g][1] for g in 1:length(p)]

function permute!(par::Param, p)
    par.xi = par.xi[p,:]
    par.beta = par.beta[p,:]
    par.mu = par.mu[p,:]
    
    par.bl_shape = par.bl_shape[p]
    par.bl_scale = par.bl_scale[p]
    par.delta = par.delta[p,:]
    
    par.S = par.S[p,:]
    par.L = par.L[p,:,:]

    par.cind = inverse_perm(p)[par.cind]
    par
end

function permute_vec(par::Param, p)
    vcat(reshape(par.xi[p,:], length(par.xi)),
         reshape(par.beta[p,:], length(par.beta)),
         reshape(par.mu[p,:], length(par.mu)),
         par.bl_shape[p], par.bl_scale[p],
         reshape(par.delta[p,:], length(par.delta)),
         reshape(par.S[p,:], length(par.S)),
         reshape(par.L[p,:,:], length(par.L)),
         inverse_perm(p)[par.cind])
end

function permute!(par::ParamSpline, p)
    par.xi = par.xi[p,:]
    par.beta = par.beta[p,:]
    par.mu = par.mu[p,:]
    
    par.w = par.w[p,:]
    par.delta = par.delta[p,:]
    
    par.S = par.S[p,:]
    par.L = par.L[p,:,:]

    par.cind = inverse_perm(p)[par.cind]
    par
end

function permute_vec(par::ParamSpline, p)
    vcat(reshape(par.xi[p,:], length(par.xi)),
         reshape(par.beta[p,:], length(par.beta)),
         reshape(par.mu[p,:], length(par.mu)),
         reshape(par.w[p,:], length(par.w)),
         reshape(par.delta[p,:], length(par.delta)),
         reshape(par.S[p,:], length(par.S)),
         reshape(par.L[p,:,:], length(par.L)),
         inverse_perm(p)[par.cind])    
end

relabel!(x::Fit, data::Data) = relabel!(x, MAP(x, data))

function relabel!(x::Tf, map::Tp) where {Tf <: Fit, Tp <: AbstractParam}
    for k in eachindex(x.chains)
        relabel!(x.chains[k], map)
    end
end

## MAP distance relabeling
function relabel!(x::Tc, map::Tp) where {Tc <: Chain, Tp <: AbstractParam}
    M = length(x)
    G = size(map.mu,1)
    
    switch = 0
    perm = collect(permutations(1:G))
    theta_min = permute_vec(map, perm[1])

    for t in 1:M
        k_min = 1
        d_min = dot(vectorize(permute_vec(x.theta[t], perm[1])), theta_min)
        
        for k in 1:length(perm)
            d = dot(vectorize(permute_vec(x.theta[t], perm[k])), theta_min)          
            if d < d_min
                d_min = d
                k_min = k
            end
        end

        if k_min != 1
            x.theta[t] = permute!(x.theta[t], perm[k_min])
            switch += 1
        end
    end
        
    @info "Labels switched in $(switch) of $(M) posterior samples!"
    
    x
end
