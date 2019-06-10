struct Prior <: AbstractPrior
    G::DiscreteDistribution
    xi::ContinuousDistribution
    beta::UnivariateDistribution
    beta_mv::MultivariateDistribution

    mu::UnivariateDistribution
    mu_mv::MultivariateDistribution

    S::UnivariateDistribution
    Sigma::ContinuousDistribution
    sigma::UnivariateDistribution
    
    bl_shape::UnivariateDistribution
    bl_scale::UnivariateDistribution
    delta::UnivariateDistribution
    delta_mv::MultivariateDistribution
end

struct PriorSpline <: AbstractPrior
    G::DiscreteDistribution
    xi::ContinuousDistribution
    beta::UnivariateDistribution
    beta_mv::MultivariateDistribution

    mu::UnivariateDistribution
    mu_mv::MultivariateDistribution

    S::UnivariateDistribution
    Sigma::ContinuousDistribution
    sigma::UnivariateDistribution
    
    w::UnivariateDistribution
    delta::UnivariateDistribution
    delta_mv::MultivariateDistribution
end

Prior(data::Data) = Prior(size(data.Xp,2), size(data.Z,3), size(data.Xl,3), size(data.Xe,2))

Prior(p=4, q=3, l=9, r=3) = Prior(Poisson(3), ## G
                                  ## xi (center pi_g at 1/G and away from 0 and 1 (Garrett/Zeger [2000], Elliot [2005], Neelono [2011])
                                  Normal(0, 3/2), 
                                  Normal(0, sqrt(10)), ## beta_uni
                                  MvNormal(diagm(0 => 10*ones(Float64, l))), ## beta_mv
                                  Normal(0, sqrt(10)), ## mu
                                  MvNormal(diagm(0 => 10*ones(Float64, q))), ## mu_mv
                                  Cauchy(0, 2.5), ## S
                                  InverseWishart(3, diagm(0 => ones(Float64, q))), ## C
                                  Cauchy(0, 2.5), ## sigma
                                  Gamma(1,1), ##Uniform(0.01, 100), ## bl_shape
                                  Gamma(1,1), ##Uniform(0.01, 100), ## bl_scale
                                  Normal(0, sqrt(10)), ## delta
                                  MvNormal(diagm(0 => 10*ones(Float64, r)))) ## delta_mv

PriorSpline(data::Data) = Prior(size(data.Xp,2), size(data.Z,3), size(data.Xl,3), size(data.Xe,2), size(data.haz_basis,2))

PriorSpline(p=4, q=3, l=9, r=3, k=9) = PriorSpline(Poisson(3), ## G
                                                   Normal(0, 3/2),
                                                   Normal(0, sqrt(10)), ## beta_uni
                                                   MvNormal(diagm(0 => 10*ones(Float64, l))), ## beta_mv
                                                   Normal(0, sqrt(10)), ## mu
                                                   MvNormal(diagm(0 => 10*ones(Float64, q))), ## mu_mv
                                                   Cauchy(0, 2.5), ## S
                                                   InverseWishart(3, diagm(0 => ones(Float64, q))), ## C
                                                   Cauchy(0, 2.5), ## sigma
                                                   Normal(0, sqrt(50)), ## w
                                                   Normal(0, sqrt(10)), ## delta
                                                   MvNormal(diagm(0 => 10*ones(Float64, r)))) ## delta_mv
