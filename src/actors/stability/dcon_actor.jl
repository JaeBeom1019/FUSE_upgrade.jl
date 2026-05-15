import NCDatasets
import Random

export Hmode_pressure_profile, Hmode_current_profile

#= ========== =#
#  ActorDCON  #
#= ========== =#
Base.@kwdef mutable struct FUSEparameters__ActorDCON{T<:Real} <: ParametersActor{T}
    _parent::WeakRef = WeakRef(nothing)
    _name::Symbol = :ActorDCON
    _time::Float64 = NaN
    #== actor parameters ==#
    dcon_n::Entry{Vector{Int}} = Entry{Vector{Int}}("-", "Toroidal mode numbers to run DCON"; default=[1])
    rdcon_n::Entry{Vector{Int}} = Entry{Vector{Int}}("-", "Toroidal mode numbers to run resistive DCON"; default=Int[])
    stride_n::Entry{Vector{Int}} = Entry{Vector{Int}}("-", "Toroidal mode numbers to run STRIDE"; default=Int[])
    ballooning::Entry{Bool} = Entry{Bool}("-", "Run DCON ballooning-mode criterion"; default=false)
    wall_flag::Entry{Bool} = Entry{Bool}("-", "Use IMAS wall geometry when true, simple DCON wall when false"; default=false)
    wall_parameter::Entry{T} = Entry{T}("-", "DCON simple-wall shape parameter"; default=20.0)
    executable_dir::Entry{String} = Entry{String}("-", "Directory containing dcon, rdcon, and stride executables"; default="/home/aspire1019/code/GPEC/bin")
    workdir::Entry{String} = Entry{String}("-", "Working directory. Empty string creates a temporary directory under /tmp"; default="")
    cleardir::Entry{Bool} = Entry{Bool}("-", "Clean working directory after running DCON"; default=true)
    verbose::Entry{Bool} = act_common_parameters(; verbose=false)
end

mutable struct ActorDCON{D,P} <: SingleAbstractActor{D,P}
    dd::IMAS.dd{D}
    par::OverrideParameters{P,FUSEparameters__ActorDCON{P}}
    results::Union{Nothing,Dict{Symbol,Any}}
end

function ActorDCON(dd::IMAS.dd, act::ParametersAllActors; kw...)
    actor = ActorDCON(dd, act.ActorDCON; kw...)
    step(actor)
    finalize(actor)
    return actor
end

function ActorDCON(dd::IMAS.dd{D}, par::FUSEparameters__ActorDCON{P}; kw...) where {D<:Real,P<:Real}
    logging_actor_init(ActorDCON)
    par = OverrideParameters(par; kw...)
    return ActorDCON(dd, par, nothing)
end

function _dcon_workdir(par)
    if isempty(par.workdir)
        working_directory = joinpath(tempdir(), "fuse_dcon_" * Random.randstring(12))
        mkpath(working_directory)
        return working_directory
    else
        mkpath(par.workdir)
        return abspath(par.workdir)
    end
end

function _required_dcon_executables(par)
    names = String[]
    (!isempty(par.dcon_n) || par.ballooning) && push!(names, "dcon")
    !isempty(par.rdcon_n) && push!(names, "rdcon")
    !isempty(par.stride_n) && push!(names, "stride")
    return unique(names)
end

function _step(actor::ActorDCON)
    dd = actor.dd
    par = actor.par

    δW = Float64[]
    Δ_rdcon = Float64[]
    Δ_stride = Float64[]
    ballooning = Float64[]

    ishape = par.wall_flag ? 42 : 6
    working_directory = _dcon_workdir(par)

    move_executable(par.executable_dir, working_directory; exe_names=_required_dcon_executables(par))
    save_gfile(dd, working_directory)
    write_equil_in(working_directory)
    write_vac_in(Float64(par.wall_parameter), ishape, working_directory)

    if ishape == 42
        write_wall_geo_in(dd, working_directory)
        par.verbose && @info "Calculating DCON with IMAS wall geometry (ishape=42)"
    else
        par.verbose && @info "Calculating DCON with simple wall parameter (ishape=6)"
    end

    for mode_num in par.dcon_n
        write_dcon_in(mode_num, working_directory)
        run_code(joinpath(working_directory, "dcon");
            log_path=joinpath(working_directory, "dcon_n$(mode_num).log"),
            working_dir=working_directory)
        push!(δW, load_dcon_output(mode_num, working_directory))
    end

    for mode_num in par.rdcon_n
        write_rdcon_in(mode_num, working_directory)
        run_code(joinpath(working_directory, "rdcon");
            log_path=joinpath(working_directory, "rdcon_n$(mode_num).log"),
            working_dir=working_directory)
        push!(Δ_rdcon, load_rdcon_output(mode_num, working_directory))
    end

    for mode_num in par.stride_n
        write_stride_in(mode_num, working_directory)
        run_code(joinpath(working_directory, "stride");
            log_path=joinpath(working_directory, "stride_n$(mode_num).log"),
            working_dir=working_directory)
        push!(Δ_stride, load_stride_output(mode_num, working_directory))
    end

    if par.ballooning
        par.verbose && @info "Running DCON ballooning criterion"
        write_dcon_in(1, working_directory)
        run_code(joinpath(working_directory, "dcon");
            log_path=joinpath(working_directory, "dcon_ballooning.log"),
            working_dir=working_directory)
        push!(ballooning, load_ballooning_output(working_directory))
    end

    actor.results = Dict{Symbol,Any}(
        :δW => δW,
        :Δ_rdcon => Δ_rdcon,
        :Δ_stride => Δ_stride,
        :ballooning => ballooning,
        :working_directory => working_directory)

    if par.cleardir
        rm(working_directory; force=true, recursive=true)
    end

    return actor
end

function _finalize(actor::ActorDCON)
    actor.results === nothing && (@warn "No results to finalize in ActorDCON"; return actor)

    dd = actor.dd
    par = actor.par
    results = actor.results
    mhd = resize!(dd.mhd_linear.time_slice; wipe=false)

    for (i, n) in enumerate(par.dcon_n)
        mode = resize!(mhd.toroidal_mode, "perturbation_type.name" => "DCON δW", "n_tor" => n)
        mode.perturbation_type.description = "Ideal MHD stability, δW from DCON"
        mode.stability_metric = results[:δW][i]
    end

    for (i, n) in enumerate(par.rdcon_n)
        mode = resize!(mhd.toroidal_mode, "perturbation_type.name" => "RDCON Δ'", "n_tor" => n)
        mode.perturbation_type.description = "Resistive MHD stability, Δ' from RDCON"
        mode.stability_metric = results[:Δ_rdcon][i]
    end

    for (i, n) in enumerate(par.stride_n)
        mode = resize!(mhd.toroidal_mode, "perturbation_type.name" => "STRIDE Δ'", "n_tor" => n)
        mode.perturbation_type.description = "Resistive MHD stability, Δ' from STRIDE"
        mode.stability_metric = results[:Δ_stride][i]
    end

    if par.ballooning && !isempty(results[:ballooning])
        mode = resize!(mhd.toroidal_mode, "perturbation_type.name" => "DCON ballooning", "n_tor" => 0)
        mode.perturbation_type.description = "Stable for ballooning mode when metric is Inf"
        mode.stability_metric = results[:ballooning][1]
    end

    actor.results = nothing
    return actor
end

function run_code(exe_path::AbstractString; log_path::AbstractString="output.log", working_dir::AbstractString=pwd())
    isfile(exe_path) || error("Executable not found: $exe_path")
    library_paths = _dcon_runtime_library_paths()
    ld_library_path = isempty(library_paths) ? get(ENV, "LD_LIBRARY_PATH", "") : join(vcat(library_paths, get(ENV, "LD_LIBRARY_PATH", "")), ":")
    env_limited = merge(ENV, Dict(
        "OMP_NUM_THREADS" => "1",
        "OPENBLAS_NUM_THREADS" => "1",
        "MKL_NUM_THREADS" => "1",
        "NUMEXPR_NUM_THREADS" => "1",
        "LD_LIBRARY_PATH" => ld_library_path))

    open(log_path, "w") do io
        Base.run(pipeline(Cmd(`$exe_path`; env=env_limited, dir=working_dir), stdout=io, stderr=io))
    end
    return nothing
end

function _dcon_runtime_library_paths()
    candidates = [
        "/opt/intel/oneapi/mkl/2025.3/lib",
        "/opt/intel/oneapi/2025.3/lib",
        "/APP/compiler/intel/oneapi/mkl/2025.0/lib",
        "/APP/enhpc/compiler/intel/v2025/oneapi/mkl/2025.0/lib"]
    return [path for path in candidates if isfile(joinpath(path, "libmkl_rt.so.2"))]
end

function move_executable(src_dir::AbstractString, working_directory::AbstractString; exe_names::AbstractVector{<:AbstractString}=["dcon", "stride", "rdcon"])
    mkpath(working_directory)
    for name in exe_names
        src_path = joinpath(src_dir, name)
        dest_path = joinpath(working_directory, name)
        isfile(src_path) || error("Required DCON executable not found: $src_path")
        cp(src_path, dest_path; force=true)
        chmod(dest_path, filemode(dest_path) | 0o111)
    end
    return nothing
end

function save_gfile(dd::IMAS.dd, working_dir::AbstractString=".")
    file_path = joinpath(working_dir, "geqdsk")
    g = CHEASE.EFIT.imas2geqdsk(dd)
    CHEASE.EFIT.writeg(g[1], file_path)
    return file_path
end

function write_equil_in(working_dir::AbstractString=".")
    content = """
&EQUIL_CONTROL
    eq_type="efit"
    eq_filename="geqdsk"

    jac_type="hamada"
    power_bp=0
    power_b=0
    power_r=0

    grid_type="ldp"
    psilow=1e-2
    psihigh=0.994
    mpsi=128
    mtheta=256
    newq0=0
    etol = 1e-7
    use_classic_splines = f

    input_only=f
    use_galgrid=t
/
&EQUIL_OUTPUT
    gse_flag=f
    out_eq_1d=f
    bin_eq_1d=f
    out_eq_2d=f
    bin_eq_2d=t
    out_2d=f
    bin_2d=f
    dump_flag=f
/
"""
    file_path = joinpath(working_dir, "equil.in")
    write(file_path, content)
    return file_path
end

function write_dcon_in(mode_number::Int, working_dir::AbstractString=".")
    content = """
&DCON_CONTROL
    bal_flag=t
    mat_flag=t
    ode_flag=t
    vac_flag=t
    mer_flag=t

    sas_flag=t
    dmlim=0.2
    psiedge=0.988
    qlow=1.02
    qhigh=20.2
    sing_start=0
    reform_eq_with_psilim=f

    nn=$(mode_number)
    delta_mlow=8
    delta_mhigh=8
    delta_mband=0
    mthvac=128
    thmax0=1

    kin_flag = f
    con_flag = t
    kinfac1 = 1.0
    kinfac2 = 1.0
    kingridtype = 0
    passing_flag = t
    trapped_flag = t
    ktanh_flag = f
    ktc = 0.1
    ktw = 50.0
    ion_flag = t
    electron_flag = f
    dcon_kin_threads=0

    tol_nr=1e-6
    tol_r=1e-7
    crossover=1e-2
    singfac_min=1e-4
    ucrit=1e3

    termbycross_flag = t

    use_classic_splines = f
    wv_farwall_flag = f
/
&DCON_OUTPUT
    out_fund=f
    out_ahg2msc=f
    crit_break=t
    ahb_flag=f
    msol_ahb=1
    mthsurf0=1

    bin_euler=f
    bin_vac=f
    euler_stride=1

    out_bal1=f
    bin_bal1=f
    out_bal2=f
    bin_bal2=f

    netcdf_out=t
/
"""
    file_path = joinpath(working_dir, "dcon.in")
    write(file_path, content)
    return file_path
end

function write_vac_in(wall_parameter::Float64, ishape::Int, working_dir::AbstractString=".")
    shape_a_line = ishape == 42 ? "" : "        a=$(wall_parameter)\n"
    content = """
!    **** Running DCON's input ***

&modes
    mth=480
    xiin(1:9)=7*0 1 0
    lsymz=t
    leqarcw=1

    lzio=0
    lgato=0
    lrgato=0
/

&debugs
    checkd=f
    check1=f
    check2=f
    checke=f
    checks=f
    wall=f
    lkplt=0
    verbose_timer_output = f
/

&vacdat
    ishape=$(ishape)
    aw=.05
    bw=1.5
    cw=0
    dw=0.5
    tw=.05
    nsing=500
    epsq=1e-5
    noutv=37
    idgt=6
    idot=0
    idsk=0
    delg=15.01
    delfac=1e-3
    cn0=1
/

&shape
    ipshp=0
    xpl=100
    apl=1
$(shape_a_line)    b=170
    bpl=1
    dpl=0
    r=1
    abulg=0.932
    bbulg=17.0
    tbulg=.02
    qain=2.5
/

&diagns
    lkdis=f
    ieig=0
    iloop=0
    lpsub=1

    nloop=128
    nloopr=0
    nphil=3
    nphse=1
    xofsl=-0
    ntloop=32
    aloop=.01
    bloop=1.6
    dloop=0.5
    rloop=1.0
    deloop=.001
    mx=21
    mz=21
    nph=0

    nxlpin = 6
    nzlpin = 11
    epslp = 0.02
    xlpmin = 0.7
    xlpmax = 2.7
    zlpmin = -1.5
    zlpmax = 1.5
    linterior = 2
/

&sprk
    nminus=0
    nplus=0
    mphi=16
    lwrt11=0
    civ=0.0
    sp2sgn1=1
    sp2sgn2=1
    sp2sgn3=1
    sp2sgn4=1
    sp2sgn5=1
    sp3sgn1=-1
    sp3sgn2=-1
    sp3sgn3=-1
    sp3sgn4=1
    sp3sgn5=1
    lff=0
    ff=1.6
    fv=5*1.6 1.0 3*1.6 1.6 25*1.6
/
"""
    file_path = joinpath(working_dir, "vac.in")
    write(file_path, content)
    return file_path
end

function write_rdcon_in(mode_number::Int, working_dir::AbstractString=".")
    content = """
&GAL_INPUT
    nx=256
    pfac=0.001
    gal_tol=1e-10
    dx1dx2_flag=t
    dx0=5e-4
    dx1=1.e-3
    dx2=1.e-3
    cutoff=10
    solver="LU"
    nq=6

    diagnose_integrand=f
    diagnose_map=f
    diagnose_grid=f
    diagnose_lsode=f
    ndiagnose=12
/
&GAL_OUTPUT
    interp_np=3
    restore_uh=t
    restore_us=t
    restore_ul=t
    bin_delmatch=f
    out_galsol=f
    bin_galsol=f
    b_flag=f
    coil%rpec_flag=f
/
&RDCON_CONTROL
    bal_flag=f
    mat_flag=t
    ode_flag=t
    vac_flag=t
    gal_flag=t

    sas_flag=t
    dmlim=0.2
    sing_start=0

    nn=$(mode_number)
    delta_mlow=8
    delta_mhigh=8
    delta_mband=0
    mthvac=512
    thmax0=1

    tol_nr=1e-5
    tol_r=1e-6
    crossover=1e-2
    singfac_min=1e-4
    ucrit=1e4

    cyl_flag=f
    sing1_flag=f
    sing_order=6
    sing_order_ceiling=t
    regrid_flag=f
/

&RDCON_OUTPUT
    out_ahg2msc=f
    crit_break=t
    ahb_flag=f
    msol_ahb=1
    mthsurf0=1
    bin_euler=f
    euler_stride=1
    out_bal1=f
    bin_bal1=f
    out_bal2=f
    bin_bal2=f
/

&UA_DIAGNOSE_LIST
    uad%flag=f
    uad%phase=t
/
"""
    file_path = joinpath(working_dir, "rdcon.in")
    write(file_path, content)
    return file_path
end

function write_stride_in(mode_number::Int, working_dir::AbstractString=".")
    content = """
&stride_control
    bal_flag=f
    mat_flag=t
    ode_flag=t
    vac_flag=t
    mer_flag=t

    sas_flag=t
    dmlim=0.2
    qlow=1.02
    qhigh=1e3
    sing_start=0

    nn=$(mode_number)
    delta_mlow=8
    delta_mhigh=8
    delta_mband=0
    mthvac=512
    thmax0=1

    tol_nr=1e-8
    tol_r=1e-8
    crossover=1e-2
    singfac_min=1e-4
    ucrit=1e4
    sing_order=6

    use_classic_splines=f
    use_notaknot_splines=f
/

&stride_output
    out_ahg2msc=f
    crit_break=t
    ahb_flag=f
    msol_ahb=1
    mthsurf0=1
    bin_euler=f
    euler_stride=1
    out_bal1=f
    bin_bal1=f
    out_bal2=f
    bin_bal2=f

    netcdf_out=t
/

&stride_params
    nThreads=0
    fourfit_metric_parallel=f
    vac_parallel=t

    nIntervalsTot=33
    grid_packing="singularities"
    axis_mid_pt_skew=12.0
    asymp_at_sing=t
    kill_big_soln_for_ideal_dW=f

    calc_delta_prime=t
    calc_dp_with_vac=t
    big_soln_err_tol=1e-7

    integrate_riccati=f
    riccati_bounce=t
    riccati_match_hamiltonian_evals=f
    verbose_riccati_output=t
    ric_dt=1e-8
    ric_tol=1e-6

    verbose_performance_output=f
/
"""
    file_path = joinpath(working_dir, "stride.in")
    write(file_path, content)
    return file_path
end

function write_wall_geo_in(dd::IMAS.dd, working_dir::AbstractString=".")
    filename = joinpath(working_dir, "wall_geo.in")
    length(dd.build.layer) >= 7 || error("dd.build.layer[7] is required to write DCON wall_geo.in")

    outline = dd.build.layer[7].outline
    r_coords = outline.r
    z_coords = outline.z
    length(r_coords) == length(z_coords) || error("R and Z coordinate lists must have the same number of points")
    !isempty(r_coords) || error("Cannot write wall_geo.in from empty wall outline")

    R0 = dd.equilibrium.time_slice[].boundary.geometric_axis.r
    Z0 = dd.equilibrium.time_slice[].boundary.geometric_axis.z
    nths = 128

    angles = map(zip(r_coords, z_coords)) do (r, z)
        θ_atan = atan(z - Z0, r - R0)
        θ_ccw_2pi = θ_atan < 0 ? θ_atan + 2π : θ_atan
        θ_sort = 3π / 2 - θ_ccw_2pi
        return θ_sort < 0 ? θ_sort + 2π : θ_sort
    end

    sorted_indices = sortperm(angles)
    theta_sorted = angles[sorted_indices]
    r_sorted = r_coords[sorted_indices]
    z_sorted = z_coords[sorted_indices]

    unique_indices = unique(i -> theta_sorted[i], eachindex(theta_sorted))
    theta_dedup = theta_sorted[unique_indices]
    r_dedup = r_sorted[unique_indices]
    z_dedup = z_sorted[unique_indices]

    theta_extended = [theta_dedup; theta_dedup[1] + 2π]
    r_extended = [r_dedup; r_dedup[1]]
    z_extended = [z_dedup; z_dedup[1]]

    itp_r = Interpolations.linear_interpolation(theta_extended, r_extended; extrapolation_bc=Interpolations.Flat())
    itp_z = Interpolations.linear_interpolation(theta_extended, z_extended; extrapolation_bc=Interpolations.Flat())
    theta_new = range(0, stop=2π, length=nths + 1)[1:end-1]
    r_new = itp_r.(theta_new)
    z_new = itp_z.(theta_new)

    open(filename, "w") do f
        println(f, nths + 2)
        @printf(f, "%.6f\n", R0)
        println(f, "theta         x_wall         z_wall")
        for i in 1:nths
            @printf(f, "%14.6f%14.6f%14.6f\n", theta_new[i], r_new[i], z_new[i])
        end
        @printf(f, "%14.6f%14.6f%14.6f\n", theta_new[1], r_new[1], z_new[1])
        @printf(f, "%14.6f%14.6f%14.6f\n", theta_new[2], r_new[2], z_new[2])
    end

    return filename
end

function load_dcon_output(mode_num::Int, working_dir::AbstractString)
    file_path = joinpath(working_dir, "dcon_output_n$(mode_num).nc")
    try
        NCDatasets.Dataset(file_path, "r") do ds
            return Float64(ds["W_t_eigenvalue"][1, 1])
        end
    catch e
        @warn "DCON failed for n=$(mode_num): $(sprint(showerror, e))"
        return -Inf
    end
end

function load_rdcon_output(mode_num::Int, working_dir::AbstractString)
    file_path = joinpath(working_dir, "rdcon_output_n$(mode_num).nc")
    try
        NCDatasets.Dataset(file_path, "r") do ds
            return Float64(ds["Delta_prime"][1, 1, 1])
        end
    catch e
        @warn "RDCON failed for n=$(mode_num): $(sprint(showerror, e))"
        return Inf
    end
end

function load_stride_output(mode_num::Int, working_dir::AbstractString)
    file_path = joinpath(working_dir, "stride_output_n$(mode_num).nc")
    try
        NCDatasets.Dataset(file_path, "r") do ds
            return Float64(ds["Delta_prime"][1, 1, 1])
        end
    catch e
        @warn "STRIDE failed for n=$(mode_num): $(sprint(showerror, e))"
        return Inf
    end
end

function load_ballooning_output(working_dir::AbstractString)
    file_path = joinpath(working_dir, "dcon_output_n1.nc")
    try
        NCDatasets.Dataset(file_path, "r") do ds
            ca1 = ds["ca1"][:]
            return all(x -> x >= 0, ca1) ? Inf : -Inf
        end
    catch e
        @warn "DCON ballooning failed: $(sprint(showerror, e))"
        return -Inf
    end
end

function Hmode_pressure_profile(edge, ped, core, expin, expout, widthp; rgrid::Int=129)
    w_E1 = 0.5 * widthp
    xphalf = 1.0 - w_E1
    xped = xphalf - w_E1
    pconst = 1.0 - tanh((1.0 - xphalf) / w_E1)
    a_t = 2.0 * (ped - edge) / (1.0 + tanh(1.0) - pconst)
    coretanh = 0.5 * a_t * (1.0 - tanh(-xphalf / w_E1) - pconst) + edge
    xpsi = range(0, 1, rgrid)
    val = 0.5 * a_t * (1.0 .- tanh.((xpsi .- xphalf) / w_E1) .- pconst) .+ edge
    xtoped = xpsi ./ xped
    for i in 1:rgrid
        if xtoped[i]^expin < 1.0
            val[i] += (core - coretanh) * (1.0 - xtoped[i]^expin)^expout
        end
    end
    return collect(val)
end

function Hmode_pressure_profile(; edge=0.1, ped=4.0e5, core=2.0e6, expin=1.5, expout=2.0, widthp=0.1, rgrid::Int=129)
    return Hmode_pressure_profile(edge, ped, core, expin, expout, widthp; rgrid)
end

function Hmode_current_profile(; std=0.03, ped=5e5, core=3.0e6, expin=3.0, expout=2.0, widthp=0.1, rgrid::Int=129)
    x = range(0, 1, length=rgrid)
    ped_pos = 1 - widthp
    j_core_arr = core .* (1 .- x .^ expin) .^ expout
    idx_ped = argmin(abs.(x .- ped_pos))
    j_at_ped = j_core_arr[idx_ped]
    gauss_height = ped - j_at_ped
    gaussian = gauss_height .* exp.(-0.5 .* ((x .- ped_pos) ./ std) .^ 2)
    return collect(j_core_arr .+ max.(gaussian, 0))
end

const HMODE_PRESSURE_PARAMETER_NAMES = (:edge, :ped, :core, :expin, :expout, :widthp)

function _default_hmode_pressure_bounds(dd::IMAS.dd)
    cp1d = dd.core_profiles.profiles_1d[]
    isempty(cp1d.pressure) && error("core_profiles pressure is required for H-mode pressure optimization")
    p0 = max(Float64(cp1d.pressure[1]), eps(Float64))
    edge0 = max(Float64(cp1d.pressure[end]), eps(Float64))
    return (
        edge=(0.1 * edge0, max(10.0 * edge0, 0.02 * p0)),
        ped=(0.15 * p0, 0.85 * p0),
        core=(0.5 * p0, 2.0 * p0),
        expin=(1.0, 4.5),
        expout=(1.0, 5.0),
        widthp=(0.04, 0.20))
end

function _bounds_from_namedtuple(bounds)
    lower = Float64[]
    upper = Float64[]
    for name in HMODE_PRESSURE_PARAMETER_NAMES
        lo, hi = getproperty(bounds, name)
        push!(lower, Float64(lo))
        push!(upper, Float64(hi))
    end
    return lower, upper
end

function _hmode_pressure_parameter_dict(x::AbstractVector)
    length(x) == length(HMODE_PRESSURE_PARAMETER_NAMES) || throw(DimensionMismatch("Expected $(length(HMODE_PRESSURE_PARAMETER_NAMES)) H-mode pressure parameters"))
    return Dict(name => Float64(x[k]) for (k, name) in enumerate(HMODE_PRESSURE_PARAMETER_NAMES))
end

function _hmode_pressure_from_vector(x::AbstractVector, rgrid::Int)
    pars = _hmode_pressure_parameter_dict(x)
    return Hmode_pressure_profile(
        pars[:edge],
        pars[:ped],
        pars[:core],
        pars[:expin],
        pars[:expout],
        pars[:widthp];
        rgrid)
end

function set_Hmode_pressure_profile!(dd::IMAS.dd, pressure::AbstractVector)
    cp1d = dd.core_profiles.profiles_1d[]
    n = length(cp1d.pressure)
    n > 0 || error("core_profiles pressure is empty")

    profile = if length(pressure) == n
        collect(Float64, pressure)
    else
        x0 = range(0.0, 1.0, length=length(pressure))
        x1 = range(0.0, 1.0, length=n)
        IMAS.interp1d(x0, pressure, :pchip).(x1)
    end

    cp1d.pressure = profile
    for field in (:pressure_thermal, :pressure_parallel, :pressure_perpendicular)
        try
            current = getproperty(cp1d, field)
            if length(current) == n
                setproperty!(cp1d, field, copy(profile))
            end
        catch
        end
    end
    return dd
end

function _dcon_metrics(dd::IMAS.dd)
    mhd = dd.mhd_linear.time_slice[]
    dcon = Float64[]
    ballooning = Float64[]
    for mode in mhd.toroidal_mode
        name = mode.perturbation_type.name
        if name == "DCON δW"
            push!(dcon, Float64(mode.stability_metric))
        elseif name == "DCON ballooning"
            push!(ballooning, Float64(mode.stability_metric))
        end
    end
    return dcon, ballooning
end

function evaluate_Hmode_pressure_DCON(
    dd::IMAS.dd,
    act::ParametersAllActors,
    x::AbstractVector;
    workdir::AbstractString,
    chease_free_boundary::Bool=false,
    dcon_cleardir::Bool=false,
    equilibrium_ip_from::Symbol=:pulse_schedule,
    qed_before_equilibrium::Bool=false,
    qed_qmin_desired::Float64=1.0,
    qed_equilibrium_iterations::Int=1,
    qed_ip_from::Union{Nothing,Symbol}=nothing,
    sawteeth_source_after_qed::Bool=false,
    sawteeth_flat_factor::Float64=1.0,
    verbose::Bool=false)

    dd_case = deepcopy(dd)
    act_case = deepcopy(act)
    cp1d = dd_case.core_profiles.profiles_1d[]
    pressure = _hmode_pressure_from_vector(x, length(cp1d.pressure))
    set_Hmode_pressure_profile!(dd_case, pressure)

    act_case.ActorEquilibrium.model = :CHEASE
    act_case.ActorCHEASE.free_boundary = chease_free_boundary
    act_case.ActorDCON.workdir = workdir
    act_case.ActorDCON.cleardir = dcon_cleardir
    act_case.ActorDCON.ballooning = true
    act_case.ActorDCON.verbose = verbose

    if qed_before_equilibrium
        ActorEquilibrium(dd_case, act_case; ip_from=equilibrium_ip_from)
        act_case.ActorCurrent.model = :QED
        act_case.ActorCurrent.ip_from = something(qed_ip_from, equilibrium_ip_from)
        act_case.ActorCurrent.vloop_from = :equilibrium
        act_case.ActorQED.solve_for = :ip
        act_case.ActorQED.ip_from = something(qed_ip_from, equilibrium_ip_from)
        act_case.ActorQED.vloop_from = :equilibrium
        act_case.ActorQED.qmin_desired = qed_qmin_desired
        for _ in 1:max(qed_equilibrium_iterations, 1)
            ActorCurrent(dd_case, act_case; ip_from=something(qed_ip_from, equilibrium_ip_from))
            if sawteeth_source_after_qed
                ActorSawteethSource(dd_case, act_case; qmin_desired=qed_qmin_desired, flat_factor=sawteeth_flat_factor)
            end
            ActorEquilibrium(dd_case, act_case; ip_from=equilibrium_ip_from)
        end
    else
        ActorEquilibrium(dd_case, act_case; ip_from=equilibrium_ip_from)
    end
    ActorDCON(dd_case, act_case)

    global_quantities = dd_case.equilibrium.time_slice[].global_quantities
    beta_normal = Float64(global_quantities.beta_normal)
    li_3 = try
        Float64(global_quantities.li_3)
    catch
        NaN
    end
    wmhd = try
        Float64(global_quantities.energy_mhd)
    catch
        NaN
    end
    q_profile = try
        Float64.(dd_case.equilibrium.time_slice[].profiles_1d.q)
    catch
        Float64.(dd_case.core_profiles.profiles_1d[].q)
    end
    q_axis = isempty(q_profile) ? NaN : first(q_profile)
    q_min = isempty(q_profile) ? NaN : minimum(q_profile)
    dcon, ballooning = _dcon_metrics(dd_case)
    ballooning_stable = !isempty(ballooning) && any(x -> isinf(x) && x > 0, ballooning)
    ideal_stable = !isempty(dcon) && all(x -> isfinite(x) && x >= 0.0, dcon)

    return (
        dd=dd_case,
        beta_normal=beta_normal,
        li_3=li_3,
        wmhd=wmhd,
        q_axis=q_axis,
        q_min=q_min,
        dcon=dcon,
        ballooning=ballooning,
        ideal_stable=ideal_stable,
        ballooning_stable=ballooning_stable,
        pressure=pressure)
end

function _append_hmode_pressure_result(csv_file::AbstractString, row::NamedTuple)
    header = "case,status,objective,beta_normal,li_3,wmhd,q_axis,q_min,ideal_stable,ballooning_stable,edge,ped,core,expin,expout,widthp,workdir,error\n"
    if !isfile(csv_file)
        write(csv_file, header)
    end
    function csv_field(value)
        text = string(value)
        if occursin(',', text) || occursin('"', text) || occursin('\n', text) || occursin('\r', text)
            return "\"" * replace(text, "\"" => "\"\"") * "\""
        else
            return text
        end
    end
    line = join((
        row.case,
        row.status,
        row.objective,
        row.beta_normal,
        row.li_3,
        row.wmhd,
        row.q_axis,
        row.q_min,
        row.ideal_stable,
        row.ballooning_stable,
        row.edge,
        row.ped,
        row.core,
        row.expin,
        row.expout,
        row.widthp,
        csv_field(row.workdir),
        csv_field(replace(row.error, '\n' => ' '))), ",") * "\n"
    open(csv_file, "a") do io
        write(io, line)
    end
    return csv_file
end

function _hmode_pressure_csv_max_case(csv_file::AbstractString)
    isfile(csv_file) || return 0
    max_case = 0
    for (k, line) in enumerate(eachline(csv_file))
        k == 1 && continue
        isempty(strip(line)) && continue
        value = tryparse(Int, first(split(line, ',')))
        value === nothing || (max_case = max(max_case, value))
    end
    return max_case
end

function _hmode_pressure_beta_from_objective(
    objective::Real,
    constraints::AbstractVector;
    penalty_unstable::Real,
    penalty_failed::Real)

    isfinite(objective) || return NaN
    objective >= 0.5 * penalty_failed && return NaN
    unstable = !isempty(constraints) && any(g -> !isfinite(g) || g > 0.0, constraints)
    if unstable
        objective >= 0.5 * penalty_unstable || return NaN
        return Float64(penalty_unstable - objective)
    else
        return Float64(-objective)
    end
end

function _seed_hmode_pressure_best_from_state!(
    best_feasible_beta::Base.RefValue{Float64},
    best_feasible_parameters::Base.RefValue{Dict{Symbol,Float64}},
    best_observed_beta::Base.RefValue{Float64},
    best_observed_parameters::Base.RefValue{Dict{Symbol,Float64}},
    state;
    penalty_unstable::Real,
    penalty_failed::Real)

    state === nothing && return nothing
    size(state.X, 1) == length(state.y) || return nothing

    for i in axes(state.X, 1)
        constraints = isempty(state.constraints) ? Float64[] : collect(view(state.constraints, i, :))
        beta_normal = _hmode_pressure_beta_from_objective(
            state.y[i],
            constraints;
            penalty_unstable,
            penalty_failed)
        isfinite(beta_normal) || continue

        pars = _hmode_pressure_parameter_dict(collect(view(state.X, i, :)))
        unstable = !isempty(constraints) && any(g -> !isfinite(g) || g > 0.0, constraints)
        if beta_normal > best_observed_beta[]
            best_observed_beta[] = beta_normal
            best_observed_parameters[] = copy(pars)
        end
        if !unstable && beta_normal > best_feasible_beta[]
            best_feasible_beta[] = beta_normal
            best_feasible_parameters[] = copy(pars)
        end
    end
    return nothing
end

function _evaluate_hmode_pressure_DCON_row(
    dd::IMAS.dd,
    act::ParametersAllActors,
    x::AbstractVector,
    case_id::Int,
    save_folder::AbstractString;
    chease_free_boundary::Bool=false,
    dcon_cleardir::Bool=false,
    equilibrium_ip_from::Symbol=:pulse_schedule,
    qed_before_equilibrium::Bool=false,
    qed_qmin_desired::Float64=1.0,
    qed_equilibrium_iterations::Int=1,
    qed_ip_from::Union{Nothing,Symbol}=nothing,
    sawteeth_source_after_qed::Bool=false,
    sawteeth_flat_factor::Float64=1.0,
    q_min_required::Float64=0.0,
    verbose::Bool=false,
    penalty_unstable::Float64=1e6,
    penalty_failed::Float64=1e9)

    pars = _hmode_pressure_parameter_dict(x)
    workdir = joinpath(save_folder, "case_$(lpad(case_id, 5, "0"))")
    try
        result = evaluate_Hmode_pressure_DCON(
            dd,
            act,
            x;
            workdir,
            chease_free_boundary,
            dcon_cleardir,
            equilibrium_ip_from,
            qed_before_equilibrium,
            qed_qmin_desired,
            qed_equilibrium_iterations,
            qed_ip_from,
            sawteeth_source_after_qed,
            sawteeth_flat_factor,
            verbose)
        q_ok = !isfinite(q_min_required) || q_min_required <= 0.0 || (isfinite(result.q_min) && result.q_min >= q_min_required)
        effective_ideal_stable = result.ideal_stable && q_ok
        unstable = !(effective_ideal_stable && result.ballooning_stable)
        penalty = q_ok ? (unstable ? penalty_unstable : 0.0) : penalty_failed
        objective = -result.beta_normal + penalty
        status = q_ok ? (unstable ? :unstable : :success) : :q_violation
        return (
            status=status,
            objective=objective,
            constraints=(effective_ideal_stable ? 0.0 : 1.0, result.ballooning_stable ? 0.0 : 1.0),
            beta_normal=result.beta_normal,
            li_3=result.li_3,
            wmhd=result.wmhd,
            unstable=unstable,
            pars=pars,
            row=(
                case=case_id,
                status=String(status),
                objective=objective,
                beta_normal=result.beta_normal,
                li_3=result.li_3,
                wmhd=result.wmhd,
                q_axis=result.q_axis,
                q_min=result.q_min,
                ideal_stable=effective_ideal_stable,
                ballooning_stable=result.ballooning_stable,
                edge=pars[:edge],
                ped=pars[:ped],
                core=pars[:core],
                expin=pars[:expin],
                expout=pars[:expout],
                widthp=pars[:widthp],
                workdir=workdir,
                error=""))
    catch e
        isa(e, InterruptException) && rethrow(e)
        return (
            status=:fail,
            objective=penalty_failed,
            constraints=(1.0, 1.0),
            beta_normal=NaN,
            li_3=NaN,
            wmhd=NaN,
            unstable=true,
            pars=pars,
            row=(
                case=case_id,
                status="fail",
                objective=penalty_failed,
                beta_normal=NaN,
                li_3=NaN,
                wmhd=NaN,
                q_axis=NaN,
                q_min=NaN,
                ideal_stable=false,
                ballooning_stable=false,
                edge=pars[:edge],
                ped=pars[:ped],
                core=pars[:core],
                expin=pars[:expin],
                expout=pars[:expout],
                widthp=pars[:widthp],
                workdir=workdir,
                error=sprint(showerror, e)))
    end
end

function optimize_Hmode_pressure_DCON(
    dd::IMAS.dd,
    act::ParametersAllActors;
    parameter_bounds=_default_hmode_pressure_bounds(dd),
    initial_samples::Int=4,
    iterations::Int=6,
    batch_size::Int=1,
    n_candidates::Int=200,
    rng_seed::Int=1,
    acquisition::Symbol=:expected_improvement,
    penalty_unstable::Float64=1e6,
    penalty_failed::Float64=1e9,
    save_folder::AbstractString=joinpath(tempdir(), "fuse_dcon_hmode_bo_" * Random.randstring(12)),
    chease_free_boundary::Bool=false,
    dcon_cleardir::Bool=false,
    equilibrium_ip_from::Symbol=:pulse_schedule,
    qed_before_equilibrium::Bool=false,
    qed_qmin_desired::Float64=1.0,
    qed_equilibrium_iterations::Int=1,
    qed_ip_from::Union{Nothing,Symbol}=nothing,
    sawteeth_source_after_qed::Bool=false,
    sawteeth_flat_factor::Float64=1.0,
    q_min_required::Float64=0.0,
    threaded::Bool=Threads.nthreads() > 1,
    distributed::Bool=false,
    distributed_workers::Vector{Int}=Distributed.workers(),
    continue_state=nothing,
    verbose::Bool=false,
    kw...)

    mkpath(save_folder)
    lowerbounds, upperbounds = _bounds_from_namedtuple(parameter_bounds)
    csv_file = joinpath(save_folder, "hmode_pressure_dcon_results.csv")
    state_count = continue_state === nothing ? 0 : size(continue_state.X, 1)
    counter = Ref(max(state_count, _hmode_pressure_csv_max_case(csv_file)))
    best_feasible_beta = Ref(-Inf)
    best_feasible_parameters = Ref(Dict{Symbol,Float64}())
    best_observed_beta = Ref(-Inf)
    best_observed_parameters = Ref(Dict{Symbol,Float64}())
    _seed_hmode_pressure_best_from_state!(
        best_feasible_beta,
        best_feasible_parameters,
        best_observed_beta,
        best_observed_parameters,
        continue_state;
        penalty_unstable,
        penalty_failed)
    write_lock = ReentrantLock()
    best_lock = ReentrantLock()

    function append_result(row)
        lock(write_lock)
        try
            _append_hmode_pressure_result(csv_file, row)
        finally
            unlock(write_lock)
        end
        return nothing
    end

    function update_best!(pars, beta_normal, unstable)
        lock(best_lock)
        try
            if beta_normal > best_observed_beta[]
                best_observed_beta[] = beta_normal
                best_observed_parameters[] = copy(pars)
            end
            if !unstable && beta_normal > best_feasible_beta[]
                best_feasible_beta[] = beta_normal
                best_feasible_parameters[] = copy(pars)
            end
        finally
            unlock(best_lock)
        end
        return nothing
    end

    function evaluate_batch(X, state)
        n = size(X, 1)
        y = Vector{Float64}(undef, n)
        constraints = Matrix{Float64}(undef, n, 2)
        status = Vector{Symbol}(undef, n)
        case_ids = collect(counter[] .+ (1:n))
        counter[] += n

        function apply_result!(i, result)
            y[i] = result.objective
            constraints[i, 1] = result.constraints[1]
            constraints[i, 2] = result.constraints[2]
            status[i] = result.status
            isfinite(result.beta_normal) && update_best!(result.pars, result.beta_normal, result.unstable)
            append_result(result.row)
            return nothing
        end

        function evaluate_one!(i)
            case_id = case_ids[i]
            x = collect(X[i, :])
            result = _evaluate_hmode_pressure_DCON_row(
                dd,
                act,
                x,
                case_id,
                save_folder;
                chease_free_boundary,
                dcon_cleardir,
                equilibrium_ip_from,
                qed_before_equilibrium,
                qed_qmin_desired,
                qed_equilibrium_iterations,
                qed_ip_from,
                sawteeth_source_after_qed,
                sawteeth_flat_factor,
                q_min_required,
                verbose,
                penalty_unstable,
                penalty_failed)
            apply_result!(i, result)
            return nothing
        end

        if distributed && n > 1
            isempty(distributed_workers) && error("distributed=true requires at least one Distributed worker")
            inputs = [(case_ids[i], collect(X[i, :])) for i in 1:n]
            worker_pool = Distributed.WorkerPool(distributed_workers)
            results = Distributed.pmap(worker_pool, inputs) do item
                case_id, x = item
                _evaluate_hmode_pressure_DCON_row(
                    dd,
                    act,
                    x,
                    case_id,
                    save_folder;
                    chease_free_boundary,
                    dcon_cleardir,
                    equilibrium_ip_from,
                    qed_before_equilibrium,
                    qed_qmin_desired,
                    qed_equilibrium_iterations,
                    qed_ip_from,
                    sawteeth_source_after_qed,
                    sawteeth_flat_factor,
                    q_min_required,
                    verbose,
                    penalty_unstable,
                    penalty_failed)
            end
            for i in 1:n
                apply_result!(i, results[i])
            end
        elseif threaded && n > 1
            Threads.@threads for i in 1:n
                evaluate_one!(i)
            end
        else
            for i in 1:n
                evaluate_one!(i)
            end
        end

        return y, constraints, status
    end

    checkpoint_file = joinpath(save_folder, "hmode_pressure_dcon_bayes.jls")
    checkpoint_callback = state -> save_bayesian_optimization(checkpoint_file, state, nothing, nothing, nothing, nothing)
    state = bayesian_optimization_loop(
        evaluate_batch,
        lowerbounds,
        upperbounds;
        initial_samples,
        iterations,
        batch_size,
        n_candidates,
        rng_seed,
        acquisition,
        continue_state,
        checkpoint_callback,
        kw...)
    save_bayesian_optimization(checkpoint_file, state, nothing, nothing, nothing, nothing)

    return (
        state=state,
        best_parameters=best_feasible_parameters[],
        best_beta_normal=isfinite(best_feasible_beta[]) ? best_feasible_beta[] : NaN,
        best_observed_parameters=best_observed_parameters[],
        best_observed_beta_normal=isfinite(best_observed_beta[]) ? best_observed_beta[] : NaN,
        save_folder=save_folder,
        csv_file=csv_file)
end
