module MCMC

using LinearAlgebra
using Distributions

export update_mean, update_var, metropolis, adaptive_metropolis

eye(n) = diagm(0 => ones(Float64, n))

## posterior interval
pint(theta::Array{Float64,2}) = [quantile(theta[:,j], [0.025, 0.975]) for j in 1:size(theta,2)]

update_mean(t::Int, Xt::Float64, m0::Float64) = (Xt + m0*(t-1)) / t
update_mean(t::Int, Xt::Array{Float64,1}, m0::Array{Float64,1}) = @. (Xt + m0*(t-1)) / t

function update_var(t::Int, Xt::Float64, mt::Float64, m0::Float64, V0::Float64)
    return V0*(t-2)/(t-1) + t*(mt-m0)^2
end

function update_var(t::Int, Xt::Array{Float64,1}, mt::Array{Float64,1},
                   m0::Array{Float64,1}, V0::Array{Float64,2}, eps::Matrix{Float64})
    return V0.*(t-2)/(t-1) + 1/(t-1).*((t-1).*(m0 * transpose(m0)) .- t.* (mt * transpose(mt)) .+ (Xt * transpose(Xt)) + eps)
end

## single Metropolis-Hastings step
function metropolis(target_log::Function, init, proposal::Function, proposal_lpdf::Function)
    prop = proposal(init)
    q1 = proposal_lpdf(prop, init)
    q2 = proposal_lpdf(init, prop)

    r = q1 + target_log(prop) - (q2 + target_log(init))

    if log(rand(1)[1]) < r
        return prop, 1
    else
        return init, 0
    end
end

function metropolis(target_log::Function, init::T, prop::T) where T <: Real
    r = target_log(prop) - target_log(init)

    if log(rand()) < r
        return prop, 1
    else
        return init, 0
    end
end

metropolis(target_log::Function, init, proposal::Function) = metropolis(target_log, init, proposal(init))

function metropolis(target_log::Function, init::T, proposal::Function, init_loglik::T) where T <: Real
    prop = proposal(init)
    tlp = target_log(prop)
    tli = target_log(init, init_loglik)

    if log(rand()) < (tlp[1] - tli[1])
        return prop, 1, tlp[2]
    else
        return init, 0, tlp[2]
    end
end

## single adaptive Metropolis step
function adaptive_metropolis(target_log::Function, init::Vector{Float64}, V::Matrix{Float64},
    t::Int, t0::Int=2*length(init), s::Float64=(2.38)^2/length(init), beta::Float64=0.05)

    d = length(init)
    proposal = zeros(Float64, d)

    if (t <= t0) | (rand() < beta)
        proposal = rand(MvNormal(init, (0.1)^2*eye(d)/d))
    else
        proposal = rand(MvNormal(init, s.*V))
    end

    r = target_log(proposal) - target_log(init)

    if log(rand()) < r
        return proposal, 1
    else
        return init, 0
    end
end

function adaptive_metropolis(target_log::Function, init::Float64, V::Float64, t::Int, t0::Int=2, s::Float64=2.38, beta::Float64=0.05)

    proposal::Float64 = 0

    if (t <= t0) | (rand() < beta)
        proposal = rand(Normal(init, 0.1))
    else
        if(s*V == 0)
            proposal = init
        else
            proposal = rand(Normal(init, s*sqrt(V)))
        end
    end

    r = target_log(proposal) - target_log(init)

    if log(rand()) < r
        return proposal, 1
    else
        return init, 0
    end
end

end
