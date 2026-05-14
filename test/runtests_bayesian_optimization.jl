using FUSE
using Test
import Random

@testset "bayesian optimization core" begin
    lowerbounds = [0.0, -1.0]
    upperbounds = [1.0, 1.0]

    X = FUSE.latin_hypercube_sampling(lowerbounds, upperbounds, 8; rng=Random.MersenneTwister(1))
    @test size(X) == (8, 2)
    @test all(X[:, 1] .>= lowerbounds[1])
    @test all(X[:, 1] .<= upperbounds[1])
    @test all(X[:, 2] .>= lowerbounds[2])
    @test all(X[:, 2] .<= upperbounds[2])

    objective(x) = sum((x .- [0.25, 0.2]).^2)
    state = FUSE.bayesian_optimize(
        objective,
        lowerbounds,
        upperbounds;
        initial_samples=6,
        iterations=8,
        batch_size=2,
        n_candidates=1000,
        rng_seed=11)

    @test size(state.X) == (22, 2)
    @test isfinite(state.best_y)
    @test state.best_y < 0.05
    @test all(state.best_x .>= lowerbounds)
    @test all(state.best_x .<= upperbounds)

    continued = FUSE.bayesian_optimize(
        objective,
        lowerbounds,
        upperbounds;
        initial_samples=0,
        iterations=1,
        batch_size=2,
        continue_state=state,
        n_candidates=200,
        rng_seed=12)

    @test size(continued.X, 1) == 24
    @test continued.best_y <= state.best_y
end
