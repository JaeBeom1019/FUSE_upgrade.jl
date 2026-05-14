import Random
import Serialization

# ====================== #
# Bayesian optimization  #
# ====================== #

Base.@kwdef mutable struct BayesianOptimizationState
    X::Matrix{Float64} = zeros(0, 0)
    y::Vector{Float64} = Float64[]
    constraints::Matrix{Float64} = zeros(0, 0)
    status::Vector{Symbol} = Symbol[]
    lowerbounds::Vector{Float64} = Float64[]
    upperbounds::Vector{Float64} = Float64[]
    best_x::Vector{Float64} = Float64[]
    best_y::Float64 = Inf
    iteration::Int = 0
    rng_seed::Int = 1
    acquisition::Symbol = :expected_improvement
end

struct LightweightGaussianProcess
    X::Matrix{Float64}
    y::Vector{Float64}
    y_mean::Float64
    y_scale::Float64
    lengthscale::Vector{Float64}
    signal_variance::Float64
    noise_variance::Float64
    factor::LinearAlgebra.Cholesky{Float64,Matrix{Float64}}
    alpha::Vector{Float64}
end

function _bounds_vectors(bounds::AbstractMatrix)
    if size(bounds, 1) == 2
        lowerbounds = collect(Float64, bounds[1, :])
        upperbounds = collect(Float64, bounds[2, :])
    elseif size(bounds, 2) == 2
        lowerbounds = collect(Float64, bounds[:, 1])
        upperbounds = collect(Float64, bounds[:, 2])
    else
        error("Bounds must be a 2 x D or D x 2 matrix")
    end
    _validate_bounds(lowerbounds, upperbounds)
    return lowerbounds, upperbounds
end

function _validate_bounds(lowerbounds::AbstractVector, upperbounds::AbstractVector)
    length(lowerbounds) == length(upperbounds) || throw(DimensionMismatch("lowerbounds and upperbounds must have the same length"))
    isempty(lowerbounds) && error("Bayesian optimization requires at least one optimization variable")
    all(isfinite, lowerbounds) || error("All lower bounds must be finite")
    all(isfinite, upperbounds) || error("All upper bounds must be finite")
    all(upperbounds .> lowerbounds) || error("Every upper bound must be larger than its lower bound")
    return nothing
end

function _clamp_to_bounds!(X::AbstractMatrix, lowerbounds::AbstractVector, upperbounds::AbstractVector)
    for j in axes(X, 2)
        X[:, j] .= clamp.(X[:, j], lowerbounds[j], upperbounds[j])
    end
    return X
end

function _scale_X(X::AbstractMatrix, lowerbounds::AbstractVector, upperbounds::AbstractVector)
    return (X .- reshape(lowerbounds, 1, :)) ./ reshape(upperbounds .- lowerbounds, 1, :)
end

function _unscale_X(Xscaled::AbstractMatrix, lowerbounds::AbstractVector, upperbounds::AbstractVector)
    return Xscaled .* reshape(upperbounds .- lowerbounds, 1, :) .+ reshape(lowerbounds, 1, :)
end

function _finite_mean(values::AbstractVector{<:Real})
    finite_values = Float64[v for v in values if isfinite(v)]
    isempty(finite_values) && return Inf
    return sum(finite_values) / length(finite_values)
end

function _finite_std(values::AbstractVector{<:Real}, mean_value::Real)
    finite_values = Float64[v for v in values if isfinite(v)]
    length(finite_values) <= 1 && return 1.0
    var = sum((v - mean_value)^2 for v in finite_values) / (length(finite_values) - 1)
    return max(sqrt(var), eps(Float64))
end

"""
    latin_hypercube_sampling(lowerbounds, upperbounds, n; rng=Random.default_rng())

Returns an `n x D` matrix sampled within the box bounds.
"""
function latin_hypercube_sampling(
    lowerbounds::AbstractVector,
    upperbounds::AbstractVector,
    n::Integer;
    rng::Random.AbstractRNG=Random.default_rng())

    _validate_bounds(lowerbounds, upperbounds)
    n > 0 || error("Number of Latin-hypercube samples must be positive")

    D = length(lowerbounds)
    X = Matrix{Float64}(undef, n, D)
    for j in 1:D
        bins = ((0:n-1) .+ Random.rand(rng, n)) ./ n
        X[:, j] .= lowerbounds[j] .+ Random.shuffle!(rng, bins) .* (upperbounds[j] - lowerbounds[j])
    end
    return X
end

function _kernel_value(x::AbstractVector, y::AbstractVector, lengthscale::AbstractVector, signal_variance::Real)
    r2 = zero(Float64)
    @inbounds for j in eachindex(x)
        d = (x[j] - y[j]) / lengthscale[j]
        r2 += d * d
    end
    return signal_variance * exp(-0.5 * r2)
end

function _kernel_matrix(X::AbstractMatrix, lengthscale::AbstractVector, signal_variance::Real)
    n = size(X, 1)
    K = Matrix{Float64}(undef, n, n)
    @inbounds for i in 1:n
        K[i, i] = signal_variance
        xi = view(X, i, :)
        for j in i+1:n
            kij = _kernel_value(xi, view(X, j, :), lengthscale, signal_variance)
            K[i, j] = kij
            K[j, i] = kij
        end
    end
    return K
end

function fit_lightweight_gp(
    X::AbstractMatrix,
    y::AbstractVector;
    lowerbounds::AbstractVector,
    upperbounds::AbstractVector,
    lengthscale::Union{Real,AbstractVector}=0.35,
    signal_variance::Real=1.0,
    noise_variance::Real=1e-6)

    size(X, 1) == length(y) || throw(DimensionMismatch("X and y must have the same number of observations"))
    finite = findall(isfinite, y)
    length(finite) >= 2 || error("At least two finite observations are required to fit the GP")

    Xfinite = Matrix{Float64}(X[finite, :])
    yfinite = Float64[y[k] for k in finite]
    Xscaled = _scale_X(Xfinite, lowerbounds, upperbounds)

    y_mean = _finite_mean(yfinite)
    y_scale = _finite_std(yfinite, y_mean)
    yscaled = (yfinite .- y_mean) ./ y_scale

    D = size(Xscaled, 2)
    ls = lengthscale isa Real ? fill(Float64(lengthscale), D) : collect(Float64, lengthscale)
    length(ls) == D || throw(DimensionMismatch("lengthscale must be scalar or have one entry per dimension"))
    ls .= max.(ls, 1e-6)

    K = _kernel_matrix(Xscaled, ls, Float64(signal_variance))
    K[LinearAlgebra.diagind(K)] .+= Float64(noise_variance)

    factor = nothing
    jitter = 1e-10
    for _ in 1:8
        factor = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(K); check=false)
        LinearAlgebra.issuccess(factor) && break
        K[LinearAlgebra.diagind(K)] .+= jitter
        jitter *= 10.0
    end
    factor === nothing || LinearAlgebra.issuccess(factor) || error("Failed to factor Gaussian-process covariance matrix")

    alpha = factor \ yscaled
    return LightweightGaussianProcess(Xscaled, yscaled, y_mean, y_scale, ls, Float64(signal_variance), Float64(noise_variance), factor, alpha)
end

function predict(model::LightweightGaussianProcess, x::AbstractVector)
    n = size(model.X, 1)
    k = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        k[i] = _kernel_value(view(model.X, i, :), x, model.lengthscale, model.signal_variance)
    end

    mu_scaled = LinearAlgebra.dot(k, model.alpha)
    solved = model.factor \ k
    var_scaled = max(model.signal_variance - LinearAlgebra.dot(k, solved), 0.0)

    mu = model.y_mean + model.y_scale * mu_scaled
    var = max(model.y_scale^2 * var_scaled, 0.0)
    return mu, var
end

_normal_pdf(z::Real) = exp(-0.5 * z * z) / sqrt(2pi)
function _normal_cdf(z::Real)
    # Abramowitz-Stegun approximation, enough for acquisition ranking.
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    cdf = 1.0 - _normal_pdf(z) * poly
    return z >= 0.0 ? cdf : 1.0 - cdf
end

function expected_improvement(best_y::Real, mu::Real, variance::Real)
    sigma = sqrt(max(variance, 0.0))
    improvement = best_y - mu
    if sigma <= 10eps(Float64)
        return max(improvement, 0.0)
    end
    z = improvement / sigma
    return improvement * _normal_cdf(z) + sigma * _normal_pdf(z)
end

function _acquisition_score(acquisition::Symbol, model::LightweightGaussianProcess, xscaled::AbstractVector, best_y::Real, exploration_weight::Real)
    mu, variance = predict(model, xscaled)
    if acquisition === :expected_improvement || acquisition === :ei
        return -expected_improvement(best_y, mu, variance)
    elseif acquisition === :lower_confidence_bound || acquisition === :lcb || acquisition === :upper_confidence_bound || acquisition === :ucb
        return mu - exploration_weight * sqrt(max(variance, 0.0))
    elseif acquisition === :mean
        return mu
    else
        error("Unsupported acquisition `$acquisition`. Use :expected_improvement, :lower_confidence_bound, or :mean")
    end
end

function _scaled_distance_to_rows(x::AbstractVector, X::AbstractMatrix)
    isempty(X) && return Inf
    best = Inf
    @inbounds for i in axes(X, 1)
        d2 = zero(Float64)
        for j in axes(X, 2)
            d = x[j] - X[i, j]
            d2 += d * d
        end
        best = min(best, sqrt(d2))
    end
    return best
end

function initialize_bayesian_state(
    lowerbounds::AbstractVector,
    upperbounds::AbstractVector;
    rng_seed::Integer=1,
    acquisition::Symbol=:expected_improvement)

    _validate_bounds(lowerbounds, upperbounds)
    return BayesianOptimizationState(
        X=zeros(0, length(lowerbounds)),
        y=Float64[],
        constraints=zeros(0, 0),
        status=Symbol[],
        lowerbounds=collect(Float64, lowerbounds),
        upperbounds=collect(Float64, upperbounds),
        best_x=Float64[],
        best_y=Inf,
        iteration=0,
        rng_seed=Int(rng_seed),
        acquisition=acquisition)
end

function _refresh_best!(state::BayesianOptimizationState)
    finite = findall(isfinite, state.y)
    if isempty(finite)
        state.best_x = Float64[]
        state.best_y = Inf
    else
        k = finite[argmin(state.y[finite])]
        state.best_x = collect(state.X[k, :])
        state.best_y = state.y[k]
    end
    return state
end

function record_bayesian_observations!(
    state::BayesianOptimizationState,
    X::AbstractMatrix,
    y::AbstractVector;
    constraints::AbstractMatrix=zeros(size(X, 1), 0),
    status::AbstractVector{Symbol}=fill(:success, size(X, 1)))

    size(X, 1) == length(y) || throw(DimensionMismatch("X and y must have the same number of observations"))
    size(X, 1) == length(status) || throw(DimensionMismatch("X and status must have the same number of observations"))
    size(X, 1) == size(constraints, 1) || throw(DimensionMismatch("X and constraints must have the same number of observations"))

    if isempty(state.X)
        state.X = Matrix{Float64}(X)
    else
        size(state.X, 2) == size(X, 2) || throw(DimensionMismatch("Observation dimension changed"))
        state.X = vcat(state.X, Matrix{Float64}(X))
    end

    append!(state.y, Float64.(y))

    if isempty(state.constraints) && size(constraints, 2) > 0
        state.constraints = Matrix{Float64}(constraints)
    elseif size(constraints, 2) > 0
        size(state.constraints, 2) == size(constraints, 2) || throw(DimensionMismatch("Constraint dimension changed"))
        state.constraints = vcat(state.constraints, Matrix{Float64}(constraints))
    elseif isempty(state.constraints)
        state.constraints = zeros(length(state.y), 0)
    else
        state.constraints = vcat(state.constraints, zeros(size(X, 1), size(state.constraints, 2)))
    end

    append!(state.status, status)
    _refresh_best!(state)
    return state
end

function propose_bayesian_candidates(
    state::BayesianOptimizationState,
    batch_size::Integer;
    n_candidates::Integer=2000,
    acquisition::Symbol=state.acquisition,
    exploration_weight::Real=2.0,
    min_distance::Real=1e-3,
    lengthscale::Union{Real,AbstractVector}=0.35,
    signal_variance::Real=1.0,
    noise_variance::Real=1e-6,
    rng::Random.AbstractRNG=Random.default_rng())

    batch_size > 0 || error("batch_size must be positive")
    _validate_bounds(state.lowerbounds, state.upperbounds)
    D = length(state.lowerbounds)

    finite = findall(isfinite, state.y)
    if length(finite) < 2
        return latin_hypercube_sampling(state.lowerbounds, state.upperbounds, batch_size; rng)
    end

    model = fit_lightweight_gp(
        state.X,
        state.y;
        lowerbounds=state.lowerbounds,
        upperbounds=state.upperbounds,
        lengthscale,
        signal_variance,
        noise_variance)

    pool_size = max(n_candidates, 10 * batch_size)
    pool = latin_hypercube_sampling(fill(0.0, D), fill(1.0, D), pool_size; rng)
    best_y = state.best_y
    scores = [_acquisition_score(acquisition, model, view(pool, i, :), best_y, exploration_weight) for i in axes(pool, 1)]
    order = sortperm(scores)

    existing_scaled = _scale_X(state.X, state.lowerbounds, state.upperbounds)
    selected_scaled = zeros(0, D)
    selected = Matrix{Float64}(undef, 0, D)
    distance = Float64(min_distance)

    while size(selected, 1) < batch_size
        for idx in order
            x = view(pool, idx, :)
            if _scaled_distance_to_rows(x, existing_scaled) < distance
                continue
            end
            if _scaled_distance_to_rows(x, selected_scaled) < distance
                continue
            end
            selected_scaled = vcat(selected_scaled, reshape(collect(x), 1, :))
            selected = _unscale_X(selected_scaled, state.lowerbounds, state.upperbounds)
            size(selected, 1) >= batch_size && break
        end
        if size(selected, 1) < batch_size
            if distance <= 1e-9
                filler = latin_hypercube_sampling(state.lowerbounds, state.upperbounds, batch_size - size(selected, 1); rng)
                selected = vcat(selected, filler)
                break
            end
            distance *= 0.1
        end
    end

    _clamp_to_bounds!(selected, state.lowerbounds, state.upperbounds)
    return selected[1:batch_size, :]
end

function bayesian_scores(
    F::AbstractMatrix,
    G::AbstractMatrix=zeros(size(F, 1), 0);
    constraint_penalty::Real=1e6)

    size(F, 2) == 1 || error("Bayesian optimization currently expects exactly one scalar objective")
    size(G, 1) == size(F, 1) || throw(DimensionMismatch("F and G must have the same number of rows"))

    scores = Vector{Float64}(undef, size(F, 1))
    for i in axes(F, 1)
        f = F[i, 1]
        violation = size(G, 2) == 0 ? 0.0 : sum(max(g, 0.0) for g in view(G, i, :) if isfinite(g))
        scores[i] = isfinite(f) ? f + constraint_penalty * violation : Inf
        if size(G, 2) > 0 && any(!isfinite, view(G, i, :))
            scores[i] = Inf
        end
    end
    return scores
end

function bayesian_optimization_loop(
    evaluate_batch::Function,
    lowerbounds::AbstractVector,
    upperbounds::AbstractVector;
    initial_samples::Integer=max(2 * length(lowerbounds) + 1, 1),
    iterations::Integer=10,
    batch_size::Integer=1,
    continue_state::Union{Nothing,BayesianOptimizationState}=nothing,
    acquisition::Symbol=:expected_improvement,
    n_candidates::Integer=2000,
    exploration_weight::Real=2.0,
    min_distance::Real=1e-3,
    lengthscale::Union{Real,AbstractVector}=0.35,
    signal_variance::Real=1.0,
    noise_variance::Real=1e-6,
    rng_seed::Integer=1,
    checkpoint_callback::Function=(state -> nothing))

    _validate_bounds(lowerbounds, upperbounds)
    initial_samples >= 0 || error("initial_samples must be non-negative")
    iterations >= 0 || error("iterations must be non-negative")
    batch_size > 0 || error("batch_size must be positive")

    state = continue_state === nothing ? initialize_bayesian_state(lowerbounds, upperbounds; rng_seed, acquisition) : continue_state
    isempty(state.lowerbounds) && (state.lowerbounds = collect(Float64, lowerbounds))
    isempty(state.upperbounds) && (state.upperbounds = collect(Float64, upperbounds))
    _validate_bounds(state.lowerbounds, state.upperbounds)

    rng = Random.MersenneTwister(state.rng_seed + max(size(state.X, 1), 0))

    while size(state.X, 1) < initial_samples
        n_batch = min(batch_size, initial_samples - size(state.X, 1))
        X = latin_hypercube_sampling(state.lowerbounds, state.upperbounds, n_batch; rng)
        y, constraints, status = evaluate_batch(X, state)
        record_bayesian_observations!(state, X, y; constraints, status)
        checkpoint_callback(state)
    end

    for _ in 1:iterations
        X = propose_bayesian_candidates(
            state,
            batch_size;
            n_candidates,
            acquisition,
            exploration_weight,
            min_distance,
            lengthscale,
            signal_variance,
            noise_variance,
            rng)
        y, constraints, status = evaluate_batch(X, state)
        record_bayesian_observations!(state, X, y; constraints, status)
        state.iteration += 1
        checkpoint_callback(state)
    end

    return state
end

function bayesian_optimize(
    objective::Function,
    lowerbounds::AbstractVector,
    upperbounds::AbstractVector;
    kw...)

    function evaluate_batch(X, state)
        y = Vector{Float64}(undef, size(X, 1))
        status = Vector{Symbol}(undef, size(X, 1))
        for i in axes(X, 1)
            try
                value = objective(collect(X[i, :]))
                y[i] = isfinite(value) ? Float64(value) : Inf
                status[i] = isfinite(y[i]) ? :success : :fail
            catch e
                isa(e, InterruptException) && rethrow(e)
                y[i] = Inf
                status[i] = :fail
            end
        end
        return y, zeros(size(X, 1), 0), status
    end

    return bayesian_optimization_loop(evaluate_batch, lowerbounds, upperbounds; kw...)
end

function save_bayesian_optimization(
    filename::AbstractString,
    state::BayesianOptimizationState,
    ini=nothing,
    act=nothing,
    objective_function=nothing,
    constraint_functions=nothing)

    data = Dict(
        "state" => state,
        "ini" => ini,
        "act" => act,
        "objective_function" => objective_function,
        "constraint_functions" => constraint_functions)
    open(filename, "w") do io
        return Serialization.serialize(io, data)
    end
end

function load_bayesian_optimization(filename::AbstractString)
    data = open(filename, "r") do io
        return Serialization.deserialize(io)
    end
    return (
        state=data["state"],
        ini=get(data, "ini", nothing),
        act=get(data, "act", nothing),
        objective_function=get(data, "objective_function", nothing),
        constraint_functions=get(data, "constraint_functions", nothing))
end
