## TODO:
##  - general cleanup (API...)
## - remove RJProp

module JLCM

using ProgressMeter
using ArgCheck
using DataFrames
using MCMCDiagnostics
using Statistics
using StatsFuns
using StatsBase
using Parameters
using LinearAlgebra
using Roots
using DataFrames
using Distributions
using Combinatorics
using Distributed
using RCall
using SharedArrays
using DynamicHMC
using TransformVariables
using MLDataUtils

using LKJ
using SplineHazard
import SplineHazard.Spline

include("MCMC.jl")
using .MCMC

include("NUTS.jl")
using .NUTSGibbs

import Base: show, size, length, summary

export show, size, length, summary

abstract type AbstractData end
abstract type AbstractParam end
abstract type AbstractFit end
abstract type AbstractPrior end

include("data.jl")
include("spline.jl")
include("param.jl")
include("prior.jl")
include("sampling.jl")
include("mcmc_chains.jl")
include("relabel.jl")
include("summary.jl")
include("cp.jl")
include("survival.jl")
include("predict.jl")
include("ic.jl")
include("xval.jl")
include("nuts_surv.jl")
include("nuts_cov.jl")
include("nuts_mlogit.jl")
include("loglik.jl")

end
