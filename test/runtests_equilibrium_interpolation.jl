using FUSE
using Test

@testset "equilibrium profile axis interpolation" begin
    psi_norm = collect(range(0.0, 1.0, length=129))
    rho_pol_norm_sqrt = sqrt.(psi_norm)
    pressure = (1.0 .- psi_norm).^2

    legacy_index = psi_norm .> 0.02
    legacy_itp = FUSE.IMAS.interp1d(
        vcat(-reverse(rho_pol_norm_sqrt[legacy_index]), rho_pol_norm_sqrt[legacy_index]),
        vcat(reverse(pressure[legacy_index]), pressure[legacy_index]),
        :pchip)

    fixed_itp = FUSE._axis_regularized_profile_interpolant(rho_pol_norm_sqrt, pressure, :pressure)

    legacy_pressure = legacy_itp.(rho_pol_norm_sqrt)
    fixed_pressure = fixed_itp.(rho_pol_norm_sqrt)

    @test maximum(abs.(diff(legacy_pressure[1:4]))) < 1e-10
    @test maximum(abs.(diff(fixed_pressure[1:4]))) > 1e-5
    @test fixed_pressure[1] == pressure[1]
    @test all(diff(fixed_pressure[1:8]) .< 0.0)

    dpressure_dpsi = [1.0e8, -2.0e4, -2.2e4]
    FUSE._regularize_axis_derivative!(dpressure_dpsi)
    @test dpressure_dpsi[1] == dpressure_dpsi[2]
end
