function mc_class_prob(x::Tf, data::Td) where {Tf <: Fit, Td <: Data}
    z = get_samples(x)
    N = length(z[1].cind)
    G = size(z[1].mu, 1)
    
    res = zeros(Int, N, G)

    for par in x.theta
        res += class_prob(par.xi, data.Xp)
    end
    
    res / length(z)
end

function mc_class_prob(x::T) where T <: Fit
    z = get_samples(x)
    N = length(z[1].cind)
    G = size(z[1].mu)[1]
    
    res = zeros(Int, N, G)

    for par in x.theta
        for i in 1:N
            res[i, par.cind[i]] += 1
        end
    end

    res / length(z)
end

## Bayes estimator of class membership
predict_class(x::T) where T <: Fit = last.(map(findmax, eachrow(mc_class_prob(x))))
