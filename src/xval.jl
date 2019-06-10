function xval_data(data::Data, K, strata=nothing)
    N = length(data.T)
    collect(kfolds(sample(1:N, N, replace=false), K))
end

merge_kfolds(x) = vcat(first.(x)...), vcat(last.(x)...)

## get inidices of training and validation datasets stratified by strata
##function xval_data(data::Data, K, strata=ones(Int, length(data.T)))    
    ##pool = 1:length(strata)    
    ##kf = [kfolds(sample(pool[strata .== s], sum(strata .== s)), K) for s in unique(strata)]
    ##[merge_kfolds([s[k] for s in kf]) for k in 1:K]
##end

## estimate elpd for each subject in holdout set
function elpd_holdout(x::T, data::Data) where T <: AbstractFit
    lp_h = dropdims(loglik_array(x, data), dims=(2))
    map(logsumexp, eachcol(lp_h)) .- log(length(x.theta))
end

## cross-validation estimate of expected pointwise log predictive density
function xval(x::Vector{T}, data::Data, subsets) where T <: AbstractFit ##=xval_data(data, 10)) where T <: AbstractFit
    ##elpd = SharedArray{Float64,1}((length(data.T)))
    ##elpd = zeros(Float64, length(data.T))
    
    ##"$(length(subsets))-fold cross validation..."
    elpd = @sync @distributed vcat for (train, test) in subsets
        ##init = [slice_param(y.init, train) for y in x]        
        y = merge(sampling(x[1].control, slice_data(data, train), typeof(x[1].init), x[1].prior; G=x[1].init.G)...)
        ##y = merge(sampling(x[1].control, slice_data(data, train), init, x[1].prior)...)
        elpd_holdout(y, slice_data(data, test))
        ##elpd[test] = 
    end

    sum(elpd), sqrt(length(elpd)*var(elpd)), elpd
end
