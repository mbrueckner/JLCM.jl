function create_spline(time, status::BitArray{1}; n_inner_knots=5)
    ## construct cubic B-spline with n_inner_knots equidistant inner knots at quantiles of observed failure times
    ev = time[status]
    outer = (0.0, maximum(ev))
    knots = quantile(ev, 0.0:(1.0/(n_inner_knots+1)):1.0)[2:(end-1)]
    
    ## the weights don't matter here since we are only evaluating the basis functions
    Spline.spline(outer, knots, ones(Float64, length(knots) + 4), 3)
end

 ## evaluate basis functions and integrals at each event time
eval_spline(data::Data, time::Vector{Float64}=data.time) = Spline.eval_basis(data.spline, time, time)
eval_spline(spline::Spline.Spline1D, time::Vector{Float64}) = Spline.eval_basis(spline, time, time)
