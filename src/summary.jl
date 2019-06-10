function param_fieldnames(::Type{Param}; include_subject_specific=false)
    a = [:G, :G0, :xi, :beta, :mu, :S, :L, :sigma, :bl_shape, :bl_scale, :delta]
    include_subject_specific ? vcat(a, [:b, :cind]) : a
end

function param_fieldnames(::Type{ParamSpline}; include_subject_specific=false)
    a = [:G, :G0, :xi, :beta, :mu, :S, :L, :sigma, :w, :delta]
    include_subject_specific ? vcat(a, [:b, :cind]) : a
end

function variables(param::T; include_subject_specific=false) where T <: AbstractParam
    names_str = Vector{String}(undef, 0)

    for par in param_fieldnames(typeof(param); include_subject_specific=include_subject_specific)
        prop = getproperty(param, par)
        prop_str = String(par)
        dims = size(prop)
        
        if length(dims) == 0
            names_str = vcat(names_str, "$(prop_str)")
        elseif length(dims) == 1
            names_str = vcat(names_str, ["$(prop_str)$(g)" for g in 1:dims[1]]...)
        elseif length(dims) == 2
            names_str = vcat(names_str, ["$(prop_str)$(g)$(k)" for k in 1:dims[2] for g in 1:dims[1]]...)
        else
            names_str = vcat(names_str, ["$(prop_str)$(g)$(k)$(l)" for l in 1:dims[3] for k in 1:dims[2] for g in 1:dims[1]]...)
        end
    end

    names_str ##Symbol.(names_str)
end

vectorize(a::T) where T <: Real = a
vectorize(a::T) where T <: AbstractArray = vec(a)

function vectorize(param::T; include_subject_specific=true) where T <: AbstractParam    
    vcat([vectorize(getproperty(param, par)) for par in param_fieldnames(typeof(param); include_subject_specific=include_subject_specific)]...)
end

function summary(x::T, data::Data) where T <: Fit
    relabel!(x, MAP(x, data))
    summary(x)
end

summary(x::T) where T <: Fit = summary(x.chains)
summary(x::AbstractVector{T}) where T <: Chain = get_mcmc_chains(x)
