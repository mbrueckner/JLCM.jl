struct Data{Tx<:Real} <: AbstractData
    ## entry::Vector{Float64}

    ## event times and indicators
    T::Vector{Tx}
    E::BitArray{1} ##Vector{Bool}

    ## fixed effects covariates 
    Xp::Matrix{Tx} # class-membership model
    Xl::Array{Tx,3} ##Matrix{Tx} # linear mixed model
    Xe::Matrix{Tx} # hazard model

    ## random effects covariates
    Z::Array{Tx,3} ##Matrix{Tx}

    ## repeated measurements
    Y::Matrix{Tx}
    times::Matrix{Tx} # measurement times
    n::Vector{Int} ## number of repeated measurments

    cind::Vector{Int}
    b::Matrix{Tx}

    ## spline basis functions and their integrals evaluated at each event time
    spline::Spline.Spline1D
    haz_basis::Matrix{Tx}
    cumhaz_basis::Matrix{Tx}
end

length(data::Data) = length(data.T)

## required for xval
function slice_data(data::Data, sel)
    Data(data.T[sel], data.E[sel], data.Xp[sel,:], data.Xl[sel,:,:], data.Xe[sel,:], data.Z[sel,:,:], data.Y[sel,:], data.times[sel,:],
         data.n[sel], data.cind[sel], data.b[sel,:], data.spline, data.haz_basis[sel,:], data.cumhaz_basis[sel,:])
end
