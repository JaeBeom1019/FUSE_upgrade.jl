"""
    workflow_bayesian_optimization(
        ini,
        act,
        actor_or_workflow,
        objective_function,
        constraint_functions;
        initial_samples,
        iterations,
        batch_size,
        save_folder,
        save_dd)

Batch Bayesian optimization of a FUSE actor or workflow. The objective function is
assumed to follow the existing `IMAS.ObjectiveFunction` minimize convention.
"""
function workflow_bayesian_optimization(
    ini::ParametersAllInits,
    act::ParametersAllActors,
    actor_or_workflow::Union{Type{<:AbstractActor},Function},
    objective_function::IMAS.ObjectiveFunction,
    constraint_functions::Vector{<:IMAS.ConstraintFunction}=IMAS.ConstraintFunction[];
    initial_samples::Int=0,
    iterations::Int=10,
    batch_size::Int=1,
    continue_state::Union{Nothing,BayesianOptimizationState}=nothing,
    acquisition::Symbol=:expected_improvement,
    n_candidates::Int=2000,
    exploration_weight::Float64=2.0,
    min_distance::Float64=1e-3,
    lengthscale::Union{Real,AbstractVector}=0.35,
    signal_variance::Float64=1.0,
    noise_variance::Float64=1e-6,
    constraint_penalty::Float64=1e6,
    rng_seed::Int=1,
    save_folder::AbstractString="bayesian_optimization_runs",
    save_dd::Bool=true,
    generation_offset::Int=0,
    kw...)

    println("Running on $(Distributed.nprocs()-1) worker processes")

    opt_ini = opt_parameters(ini)
    isempty(opt_ini) && error("No optimization variables found in ini. Mark parameters with `↔` bounds first.")

    println("== Actuators ==")
    for optpar in opt_ini
        println(optpar)
    end
    println()
    println("== Objective ==")
    println(objective_function)
    println()
    println("== Constraints ==")
    for cnst in constraint_functions
        println(cnst)
    end

    bounds = float_bounds(opt_ini)
    lowerbounds, upperbounds = _bounds_vectors(bounds)

    if initial_samples == 0
        initial_samples = max(2 * length(lowerbounds) + 1, batch_size)
    end

    save_folder = abspath(save_folder)
    if !isempty(save_folder)
        mkpath(save_folder)
    end

    previous_batches = continue_state === nothing ? 0 : ceil(Int, size(continue_state.X, 1) / batch_size)
    generation_offset = max(generation_offset, previous_batches)

    init_remaining = continue_state === nothing ? initial_samples : max(initial_samples - size(continue_state.X, 1), 0)
    total_batches = ceil(Int, init_remaining / batch_size) + iterations
    p = ProgressMeter.Progress(max(total_batches, 1); desc="Bayesian iteration", showspeed=true)

    objective_functions = IMAS.ObjectiveFunction[objective_function]

    function evaluate_batch(X, state)
        F, G, _ = optimization_engine(
            ini,
            act,
            actor_or_workflow,
            X,
            objective_functions,
            constraint_functions,
            save_folder,
            save_dd,
            p,
            generation_offset;
            number_of_generations=max(generation_offset + total_batches, 1),
            population_size=batch_size,
            kw...)
        y = bayesian_scores(F, G; constraint_penalty)
        status = [isfinite(v) ? :success : :fail for v in y]
        return y, G, status
    end

    checkpoint_file = joinpath(save_folder, "bayesian_results.jls")
    checkpoint_callback = state -> save_bayesian_optimization(checkpoint_file, state, ini, act, objective_function, constraint_functions)

    state = bayesian_optimization_loop(
        evaluate_batch,
        lowerbounds,
        upperbounds;
        initial_samples,
        iterations,
        batch_size,
        continue_state,
        acquisition,
        n_candidates,
        exploration_weight,
        min_distance,
        lengthscale,
        signal_variance,
        noise_variance,
        rng_seed,
        checkpoint_callback)

    save_bayesian_optimization(checkpoint_file, state, ini, act, objective_function, constraint_functions)
    return state
end

function workflow_bayesian_optimization(
    ini::ParametersAllInits,
    act::ParametersAllActors,
    actor_or_workflow::Union{Type{<:AbstractActor},Function},
    objective_functions::Vector{<:IMAS.ObjectiveFunction},
    constraint_functions::Vector{<:IMAS.ConstraintFunction}=IMAS.ConstraintFunction[];
    kw...)

    length(objective_functions) == 1 || error("Bayesian optimization currently supports exactly one objective function")
    return workflow_bayesian_optimization(ini, act, actor_or_workflow, objective_functions[1], constraint_functions; kw...)
end
