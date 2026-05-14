#= ======================== =#
#  StudyBayesianOptimizer   #
#= ======================== =#

"""
    study_parameters(::Val{:BayesianOptimizer})

Parameters for batch Bayesian optimization over `OptParameter` ranges in `ini`.
"""
function study_parameters(::Val{:BayesianOptimizer})
    return FUSEparameters__ParametersStudyBayesianOptimizer{Real}()
end

Base.@kwdef mutable struct FUSEparameters__ParametersStudyBayesianOptimizer{T<:Real} <: ParametersStudy{T}
    _parent::WeakRef = WeakRef(nothing)
    _name::Symbol = :StudyBayesianOptimizer
    # HPC related parameters
    server::Switch{String} = study_common_parameters(; server="localhost")
    n_workers::Entry{Int} = study_common_parameters(; n_workers=missing)
    file_save_mode::Switch{Symbol} = study_common_parameters(; file_save_mode=:safe_write)
    release_workers_after_run::Entry{Bool} = study_common_parameters(; release_workers_after_run=true)
    restart_workers_after_n_iterations::Entry{Int} =
        Entry{Int}("-", "Restart workers every N Bayesian iterations; 0 disables worker restarts"; default=0)
    save_dd::Entry{Bool} = study_common_parameters(; save_dd=true)
    save_folder::Entry{String} = Entry{String}("-", "Folder to save Bayesian optimization runs into")
    database_policy::Switch{Symbol} = study_common_parameters(; database_policy=:single_hdf5)
    # Bayesian optimization parameters
    initial_samples::Entry{Int} = Entry{Int}("-", "Initial Latin-hypercube samples; 0 selects 2D+1")
    number_of_iterations::Entry{Int} = Entry{Int}("-", "Number of Bayesian acquisition/update iterations")
    batch_size::Entry{Int} = Entry{Int}("-", "Number of FUSE cases evaluated per Bayesian iteration")
    acquisition::Switch{Symbol} =
        Switch{Symbol}([:expected_improvement, :lower_confidence_bound, :mean], "-", "Acquisition function"; default=:expected_improvement)
    n_candidates::Entry{Int} = Entry{Int}("-", "Candidate pool size for acquisition maximization"; default=2000)
    exploration_weight::Entry{T} = Entry{T}("-", "Exploration multiplier for lower-confidence-bound acquisition"; default=2.0)
    min_distance::Entry{T} = Entry{T}("-", "Minimum scaled distance between proposed points"; default=1e-3)
    lengthscale::Entry{T} = Entry{T}("-", "Squared-exponential GP lengthscale on normalized variables"; default=0.35)
    signal_variance::Entry{T} = Entry{T}("-", "Gaussian-process signal variance"; default=1.0)
    noise_variance::Entry{T} = Entry{T}("-", "Gaussian-process noise variance"; default=1e-6)
    constraint_penalty::Entry{T} = Entry{T}("-", "Penalty multiplier for positive constraint violation"; default=1e6)
    rng_seed::Entry{Int} = Entry{Int}("-", "Random seed"; default=1)
end

mutable struct StudyBayesianOptimizer{T<:Real} <: AbstractStudy
    sty::OverrideParameters{T,FUSEparameters__ParametersStudyBayesianOptimizer{T}}
    ini::ParametersAllInits
    act::ParametersAllActors
    constraint_functions::Vector{IMAS.ConstraintFunction}
    objective_function::IMAS.ObjectiveFunction
    state::Union{Nothing,BayesianOptimizationState}
    dataframe::Union{DataFrame,Missing}
    datafame_filtered::Union{DataFrame,Missing}
    iteration::Int
    workflow::Union{Function,Missing}
end

function StudyBayesianOptimizer(
    sty::ParametersStudy,
    ini::ParametersAllInits,
    act::ParametersAllActors,
    constraint_functions::Vector{IMAS.ConstraintFunction},
    objective_function::IMAS.ObjectiveFunction;
    kw...)

    sty = OverrideParameters(sty; kw...)
    study = StudyBayesianOptimizer(sty, ini, act, constraint_functions, objective_function, nothing, missing, missing, 0, missing)
    check_and_create_file_save_mode(sty)
    parallel_environment(sty.server, sty.n_workers)
    return study
end

function StudyBayesianOptimizer(
    sty::ParametersStudy,
    ini::ParametersAllInits,
    act::ParametersAllActors,
    objective_function::IMAS.ObjectiveFunction;
    constraint_functions::Vector{IMAS.ConstraintFunction}=IMAS.ConstraintFunction[],
    kw...)

    return StudyBayesianOptimizer(sty, ini, act, constraint_functions, objective_function; kw...)
end

function _run(study::StudyBayesianOptimizer)
    sty = study.sty

    @assert (sty.n_workers == 0 || sty.n_workers == length(Distributed.workers())) "The number of workers = $(length(Distributed.workers())) isn't the number of workers you requested = $(sty.n_workers)"
    @assert sty.batch_size > 0 "Batch size must be positive"
    @assert sty.number_of_iterations >= 0 "Number of Bayesian iterations must be non-negative"
    @assert sty.initial_samples >= 0 "Initial samples must be non-negative"

    if ismissing(study.workflow)
        study.workflow = optimization_workflow_default
    end

    if sty.restart_workers_after_n_iterations > 0
        max_iters_per_run = sty.restart_workers_after_n_iterations
        steps = Int(ceil(sty.number_of_iterations / max_iters_per_run))
        sty_bkp = deepcopy(sty)
        for i in 1:steps
            try
                println("Running $max_iters_per_run Bayesian iterations ($i / $steps)")
                iters = i == steps ? sty_bkp.number_of_iterations - max_iters_per_run * (steps - 1) : max_iters_per_run
                sty.restart_workers_after_n_iterations = 0
                sty.release_workers_after_run = false
                sty.file_save_mode = :append
                sty.number_of_iterations = iters
                run(study)
            catch e
                isa(e, InterruptException) && rethrow(e)
                @warn "error occurred in Bayesian restart step $i\n$(string(e))"
            finally
                sty.restart_workers_after_n_iterations = sty_bkp.restart_workers_after_n_iterations
                sty.release_workers_after_run = sty_bkp.release_workers_after_run
                sty.file_save_mode = sty_bkp.file_save_mode
                sty.number_of_iterations = sty_bkp.number_of_iterations
            end
        end
        Distributed.rmprocs(Distributed.workers())
        @info "released workers"
        return study
    end

    @assert !isempty(sty.save_folder) "Specify where you would like to store your optimization results in sty.save_folder"

    study.state = workflow_bayesian_optimization(
        study.ini,
        study.act,
        study.workflow,
        study.objective_function,
        study.constraint_functions;
        initial_samples=sty.initial_samples,
        iterations=sty.number_of_iterations,
        batch_size=sty.batch_size,
        continue_state=study.state,
        acquisition=sty.acquisition,
        n_candidates=sty.n_candidates,
        exploration_weight=sty.exploration_weight,
        min_distance=sty.min_distance,
        lengthscale=sty.lengthscale,
        signal_variance=sty.signal_variance,
        noise_variance=sty.noise_variance,
        constraint_penalty=sty.constraint_penalty,
        rng_seed=sty.rng_seed,
        save_folder=sty.save_folder,
        save_dd=sty.save_dd,
        generation_offset=study.iteration,
        sty.database_policy)

    study.iteration = ceil(Int, size(study.state.X, 1) / sty.batch_size)

    if study.sty.database_policy == :separate_folders
        extract_results(study)
    else
        study.dataframe = _merge_tmp_study_files(sty.save_folder; cleanup=true)
        study.datafame_filtered = filter_outputs(study.dataframe, [c.name for c in study.constraint_functions])
    end

    if sty.release_workers_after_run
        Distributed.rmprocs(Distributed.workers())
        @info "released workers"
    end

    return study
end
