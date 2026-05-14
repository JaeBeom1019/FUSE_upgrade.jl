# FUSE Equilibrium Solver Debugging and Bayesian Optimization Plan

작성 위치: `/home/aspire1019/code/FUSE_upgrade`

대상 clone: `https://github.com/JaeBeom1019/FUSE_upgrade.jl.git`

목표:

1. `ActorEquilibrium` 이후 CHEASE/TEQUILA 결과에서 core 쪽 `dpressure_dpsi` 또는 `P'` 앞쪽 몇 개 point가 거의 0으로 눌리며 stepped pressure profile처럼 보이는 원인을 규명하고 수정한다.
2. CHEASE가 첫 1-2회는 동작하다가 반복 실행 중 array 관련 에러를 내는 원인을 FUSE actor sequencing, CHEASE.jl file I/O, 입력 배열 불변조건 관점에서 분리해 잡는다.
3. 기존 genetic algorithm workflow와 같은 사용감을 유지하면서, Julia 내부에 최소 의존성 Bayesian optimization 기능을 추가하는 설계를 만든다.

이 문서는 아직 구현 patch가 아니라, 코드 검토에 기반한 실행 계획이다. 현재 실행 중인 production FUSE queue에는 영향을 주지 않도록 `/home/aspire1019/code/FUSE_upgrade` clone에서 먼저 수정/검증한다.

---

## 1. 검토한 코드 경로

### 1.1 FUSE equilibrium 공통 경로

- `src/actors/equilibrium/equilibrium_actor.jl`
- 핵심 함수:
  - `ActorEquilibrium(...)`
  - `_step(actor::ActorEquilibrium)`
  - `_finalize(actor::ActorEquilibrium)`
  - `prepare(actor::ActorEquilibrium)`

중요 흐름:

1. `_step(actor::ActorEquilibrium)`에서 solver 실행 전에 항상 `prepare(actor)`를 호출한다.
2. `prepare(actor)`가 `dd.core_profiles.profiles_1d[]` 또는 `dd.equilibrium.time_slice[].profiles_1d`에서 pressure/current profile을 가져온다.
3. `prepare(actor)`가 `dd.equilibrium.time_slice[].profiles_1d`를 새로 채운다.
4. 이후 선택된 solver인 `ActorTEQUILA`, `ActorCHEASE`, `ActorFRESCO`, `ActorEGGO` 등이 이 1D profile을 입력으로 사용한다.
5. solver finalize 후 `IMAS.flux_surfaces(eqt, fw.r, fw.z)`가 호출된다.

### 1.2 TEQUILA 경로

- `src/actors/equilibrium/tequila_actor.jl`
- `_step(actor::ActorTEQUILA)`는 `eqt1d.pressure`, `eqt1d.j_tor`를 `TEQUILA.FE(...)`로 넘긴다.
- `fixed_grid == :toroidal` 기본값이면 `eqt1d.rho_tor_norm` grid에 pressure/current를 고정한다.
- `_finalize(actor::ActorTEQUILA)`에서 `MXHEquilibrium.pressure(...)`와 `MXHEquilibrium.pressure_gradient(...)`로 output `pressure`, `dpressure_dpsi`를 다시 채운다.

### 1.3 CHEASE 경로

- `src/actors/equilibrium/chease_actor.jl`
- `_step(actor::ActorCHEASE)`는 `eqt1d.psi_norm`, `eqt1d.pressure`, `eqt1d.j_tor`에서 `rho_pol = sqrt.(psin)`을 만들고 `CHEASE.run_chease(...)`에 넘긴다.
- `_finalize(actor::ActorCHEASE)`는 `actor.chease.gfile`을 필요하면 free-boundary 변환한 뒤 `gEQDSK2IMAS(...)`로 IMAS equilibrium을 채운다.
- `gEQDSK2IMAS(...)`에서 CHEASE output `g.pres`, `g.pprime`, `g.fpol`, `g.ffprim`이 IMAS 1D profiles로 들어간다.

### 1.4 CHEASE.jl package 경로

- `~/.julia/packages/CHEASE/vUsqJ/src/CHEASE.jl`
- `~/.julia/packages/CHEASE/vUsqJ/src/CHEASE_file_IO.jl`

중요 관찰:

- `run_chease(...)`는 매 호출마다 `mktempdir()`로 새 run directory를 만든다.
- `clear_workdir=true`이면 output gfile을 읽은 뒤 temp directory를 삭제한다.
- `write_EXPEQ_file(...)`는 `rho_pol`, `pressure`, `j_tor` 길이가 같은지만 assert한다.
- `rho_pol` monotonicity, finite value, duplicate point, `[0, 1]` endpoint, pressure monotonicity, current sign consistency는 CHEASE.jl 쪽에서 강하게 검증하지 않는다.

### 1.5 기존 GA/optimization 경로

- `src/optimization.jl`
- `src/workflows/optimization_workflow.jl`
- `src/studies/multi_objective_optimization.jl`

중요 흐름:

- `opt_parameters(ini)`로 optimization variable을 찾는다.
- `float_bounds(opt_ini)`로 bounds를 만든다.
- `parameters_from_opt!(ini, x)`로 candidate vector를 `ini`에 반영한다.
- worker는 `optimization_engine(...)`에서 deepcopy된 `ini`, `act`로 `init` 또는 workflow를 실행한다.
- 성공/실패 결과는 `tmp_h5_output`, `tmp_csv_output`, `database.h5`, `extract.csv`로 저장된다.
- GA는 `Metaheuristics.ECA` 또는 `Metaheuristics.SPEA2`를 사용한다.

### 1.6 BayesianOptimization.jl upstream 구조

검토 위치: `/tmp/BayesianOptimization.jl`

중요 구조:

- `BOpt` state가 objective function, model, acquisition, optimizer, bounds, observed optimum, iteration counter를 가진다.
- 초기점은 `ScaledSobolIterator` 또는 `ScaledLHSIterator`로 만든다.
- 매 iteration:
  - model hyperparameter를 fit/update한다.
  - acquisition function을 최대화한다.
  - 새 point를 evaluate한다.
  - GP model을 update한다.
- acquisition은 `ExpectedImprovement`, `ProbabilityOfImprovement`, `UpperConfidenceBound` 등이 있다.
- 의존성은 `GaussianProcesses`, `NLopt`, `Sobol`, `ElasticArrays`, `ElasticPDMats`, `DiffResults`, `ForwardDiff`, `SpecialFunctions`, `TimerOutputs`이다.

FUSE에 그대로 의존성으로 넣기에는 `GaussianProcesses`와 `NLopt`가 추가되고, HPC worker 환경에서 precompile/manifest 충돌 리스크가 생긴다. 따라서 1차 구현은 내부 경량 BO로 가는 것이 낫다.

---

## 2. Issue 1: core `P'` flattening / stepped pressure profile

### 2.1 현상 정의

관측 현상:

- CHEASE 또는 TEQUILA를 풀고 난 뒤 equilibrium 1D profile의 core 쪽 앞 5개 정도 array point에서 `P'` 또는 `dpressure_dpsi`가 거의 0에 가깝다.
- 그 결과 pressure가 axis 부근에서 물리적으로 smooth하게 감소하지 않고 plateau 후 step처럼 보인다.

물리적으로 axis에서 정확히 `dP/dpsi = 0`이 되는 것은 이상하지 않다. 문제는 axis 단일 point가 아니라 첫 여러 grid point가 같이 0에 가까워져 pressure curvature가 사라지는 것이다.

### 2.2 1차 root-cause 가설

가장 유력한 원인은 solver 개별 코드보다 `ActorEquilibrium.prepare`의 공통 pressure/current preprocessing이다.

문제 후보:

```julia
index = cp1d.grid.psi_norm .> 0.02
rho_pol_norm_sqrt0 = vcat(-reverse(sqrt.(cp1d.grid.psi_norm[index])), sqrt.(cp1d.grid.psi_norm[index]))
j_tor0 = vcat(reverse(cp1d.j_tor[index]), cp1d.j_tor[index])
pressure0 = vcat(reverse(cp1d.pressure[index]), cp1d.pressure[index])
j_itp = IMAS.interp1d(rho_pol_norm_sqrt0, j_tor0, :pchip)
p_itp = IMAS.interp1d(rho_pol_norm_sqrt0, pressure0, :pchip)
...
eqt1d.j_tor = j_itp.(sqrt.(eqt1d.psi_norm))
eqt1d.pressure = p_itp.(sqrt.(eqt1d.psi_norm))
```

분석:

- `psi_norm <= 0.02`를 모두 제외한다.
- 이후 남은 첫 positive point를 음수 쪽으로 mirror해서 axis zero-gradient 조건을 강제로 만든다.
- grid가 coarse하거나 `psi_norm`이 nonuniform이면 `0.02` cutoff 안에 output grid의 앞 4-6개 point가 들어갈 수 있다.
- 이 경우 axis 주변 interpolation이 실제 core pressure shape가 아니라 `psi_norm > 0.02`에서 시작한 대칭 PCHIP의 외삽/보간에 의해 결정된다.
- 결과적으로 `eqt1d.pressure` 자체가 solver 입력 단계에서 이미 flat core를 가질 수 있다.
- CHEASE와 TEQUILA 모두 이 `eqt1d.pressure`를 입력으로 쓰므로, 두 solver 모두에서 같은 증상이 나올 수 있다.

### 2.3 2차 root-cause 가설

`eqt1d.psi_norm`이 `prepare`에서 명시적으로 복사되지 않는 점도 확인 대상이다.

현재 `prepare`는 아래처럼 `psi`와 `rho_tor_norm`은 명시적으로 설정하지만, `psi_norm`은 직접 설정하지 않는다.

```julia
eqt1d.psi = psi0
eqt1d.rho_tor_norm = rho_tor_norm0
eqt1d.j_tor = j_itp.(sqrt.(eqt1d.psi_norm))
eqt1d.pressure = p_itp.(sqrt.(eqt1d.psi_norm))
```

IMAS expression이 `eqt1d.psi`에서 `psi_norm`을 계산할 수 있지만, 반복 solve 직후 stale expression, previous equilibrium mapping, non-monotonic `psi`가 섞이면 interpolation sampling grid가 의도와 다를 수 있다. 이 부분은 반드시 diagnostic으로 확인한다.

### 2.4 3차 root-cause 가설

`past_time_slice` branch에서 geometric factor를 이용해 `dpressure_dpsi`와 `f_df_dpsi`를 재계산한다.

```julia
tmp = IMAS.calc_pprime_ffprim_f(psi, gm8, gm9, gm1, r0, b0; pressure, j_tor)
eqt1d.dpressure_dpsi = IMAS.interp1d(psin, tmp.dpressure_dpsi).(psin0)
eqt1d.f_df_dpsi = IMAS.interp1d(psin, tmp.f_df_dpsi).(psin0)
```

여기서도 `pressure`가 이미 flat이면 `dpressure_dpsi`는 당연히 flat하다. 또한 이 branch에서 계산한 `f`는 현재 주석 처리되어 `eqt1d.f`로 들어가지 않는다. 이것이 직접 `P'` flattening 원인이라고 단정할 수는 없지만, equilibrium consistency diagnostic에 포함해야 한다.

### 2.5 solver별 영향 경로

TEQUILA:

- `_step`에서 `P = (TEQUILA.FE(rho, eqt1d.pressure), :toroidal 또는 :poloidal)`를 만든다.
- 입력 `eqt1d.pressure`가 flat core면 `MXHEquilibrium.pressure_gradient(...)` output도 flat core가 된다.

CHEASE:

- `_step`에서 `rho_pol = sqrt.(eqt1d.psi_norm)`, `pressure = eqt1d.pressure`를 `CHEASE.run_chease(...)`로 넘긴다.
- CHEASE output `g.pprime`이 flat core인 경우, 입력 `EXPEQ`의 pressure profile이 이미 flat인지부터 확인해야 한다.

### 2.6 진단 계획

먼저 solver를 고치기 전에 `prepare -> solver step -> solver finalize`의 각 단계에서 profile을 저장한다.

추가할 diagnostic helper 후보:

- 파일 후보: `src/actors/equilibrium/equilibrium_diagnostics.jl`
- 또는 테스트/스크립트 후보: `test/equilibrium/profile_prepare_diagnostics.jl`

진단 함수 후보:

```julia
equilibrium_profile_snapshot(label, eqt1d_or_cp1d; ncore=10)
```

수집할 값:

- `psi`
- `psi_norm`
- `sqrt(psi_norm)`
- `rho_tor_norm`
- `pressure`
- finite-difference `dpressure/dpsi`
- stored `dpressure_dpsi`가 있으면 stored 값
- `j_tor`
- `q`가 있으면 `q`
- 각 array length
- finite/nonfinite count
- monotonicity check 결과
- first 10 values

진단 단계:

1. `dd.core_profiles.profiles_1d[]` 원본 snapshot.
2. `prepare(actor)` 직후 `dd.equilibrium.time_slice[].profiles_1d` snapshot.
3. solver `_step` 진입 직전 snapshot.
4. solver `_finalize` 직후 snapshot.
5. `ActorEquilibrium._finalize`에서 `IMAS.flux_surfaces(...)` 이후 snapshot.

판정 metric:

- `n_flat_core = count(abs.(dP_dpsi[1:ncore]) .< atol + rtol * maximum(abs.(dP_dpsi)))`
- `pressure_jump_ratio = maximum(abs.(diff(pressure[1:ncore]))) / max(abs(pressure[1] - pressure[end]), eps())`
- `flat_core_width = maximum(rho[flat_indices])`
- `input_output_core_gradient_ratio = norm(dP_out[2:ncore]) / norm(dP_in[2:ncore])`

합격 기준 초안:

- axis point 하나는 `dP/dpsi ~= 0` 허용.
- `rho_pol > 0.02` 이후에는 first 5 points 전체가 0 근처로 잠기면 fail.
- 원본 core pressure가 감소 profile이면 `prepare` 후 first 5 pressure가 모두 같은 값에 가까워지면 fail.
- CHEASE/TEQUILA output의 `dpressure_dpsi`가 input finite-difference gradient와 부호/스케일에서 급격히 불일치하면 fail.

### 2.7 수정 계획

#### Patch A: `prepare`에서 축 근처 hard cutoff 제거 또는 parameter화

현재 `psi_norm > 0.02` hard cutoff는 너무 강하다. 다음 중 하나로 바꾼다.

권장안 A1:

- `psi_norm == 0` axis point는 유지한다.
- axis derivative zero 조건은 첫 positive point를 제거해서 강제하지 않는다.
- 대신 axis 주변 smoothness를 위해 axis 포함 grid에 대해 shape-preserving interpolation을 수행한다.
- 필요한 경우 `rho=0`에서 derivative 0인 ghost point를 하나만 만든다.

권장안 A2:

- cutoff를 고정 `0.02`가 아니라 grid 기반으로 바꾼다.
- 예: `axis_smoothing_points = 1` 또는 `axis_smoothing_rho = min(first positive rho spacing, user threshold)`.
- default는 기존보다 훨씬 약하게 둔다.

권장안 A3:

- core pressure를 직접 mirror하지 않고, `pressure(rho)`를 axis Taylor 형태로 regularize한다.
- `P(rho) = P0 + a * rho^2 + b * rho^4` 형태를 first 몇 points에 fit해서 axis derivative만 0으로 맞춘다.
- 이 방식은 `dP/drho = 0` at axis를 만족하면서도 `dP/dpsi`가 여러 point에서 0으로 잠기는 문제를 줄인다.

우선 구현 난이도와 안정성을 고려하면 A2 또는 A3가 적합하다.

#### Patch B: `psi_norm` 명시적 설정/검증

`prepare`에서 `eqt1d.psi = psi0` 직후 다음을 명시적으로 보장한다.

- `psi0`가 monotonic인지 확인.
- `psi_norm0 = (psi0 .- psi0[1]) ./ (psi0[end] - psi0[1])`를 계산하거나 원본 `cp1d.grid.psi_norm`을 복사한다.
- `eqt1d.psi_norm` expression 사용 전 length/finite/endpoint를 확인한다.

주의:

- IMAS schema에서 `psi_norm`이 settable expression인지 확인해야 한다.
- 직접 set이 불가하면 local variable `psi_norm_eval`을 만들고 `sqrt.(psi_norm_eval)`로 interpolation을 수행한다.

#### Patch C: pressure/current input validator 추가

공통 helper 후보:

```julia
validate_equilibrium_1d_input!(eqt1d; source, repair=false)
```

검증 항목:

- `length(psi) == length(pressure) == length(j_tor) == length(rho_tor_norm)`
- all finite
- `psi` monotonic
- `psi_norm[1] == 0`, `psi_norm[end] == 1` within tolerance
- `pressure >= pressure[end]` mostly monotonic decreasing
- `pressure[end] >= 0`
- `j_tor` sign does not oscillate heavily after sign filtering
- no duplicate `rho_pol`/`rho_tor_norm`

`repair=true`이면:

- tiny negative pressure를 0 또는 separatrix pressure로 clamp.
- duplicate grid point 제거.
- sorted unique grid로 resample.
- `rho[1] = 0`, `rho[end] = 1` 강제.

#### Patch D: regression test 추가

테스트 후보:

- analytic profile: `P = P0 * (1 - psi_norm)^2 + Psep`
- 이 profile은 `dP/dpsi = -2P0(1 - psi_norm)`이므로 axis 근처에서 0이 아니라 큰 음수다.
- 단, `P(rho)` 관점에서는 axis regularity 때문에 `dP/drho = 0`일 수 있다. 따라서 `dP/dpsi`와 `dP/drho`를 혼동하지 않도록 테스트 metric을 둘로 나눈다.

테스트 pass 조건:

- `prepare` 후 `pressure`가 원본 analytic profile과 core에서 큰 오차 없이 일치한다.
- `dP/dpsi` finite difference가 first non-axis points에서 모두 0으로 붕괴하지 않는다.
- TEQUILA finalize 후 first 5 `dpressure_dpsi`가 모두 near-zero가 아니다.
- CHEASE finalize 후 `g.pprime` 기반 `dpressure_dpsi`가 모두 near-zero가 아니다.

---

## 3. Issue 2: CHEASE 반복 실행 array 에러

### 3.1 현상 정의

관측 현상:

- CHEASE가 처음 1-2회는 돌아간다.
- 이후 반복 실행 중 array shape, indexing, bounds, interpolation, GEQDSK 변환 계열로 보이는 에러가 발생한다.
- 의심: CHEASE actor script code의 solve 순서 또는 state reuse 문제.

### 3.2 FUSE actor 쪽 risk point

`ActorCHEASE`는 mutable field로 `chease::Union{Nothing,CHEASE.Chease}`를 가진다. 매 `_step`에서 새 `actor.chease = CHEASE.run_chease(...)`를 받지만, `_finalize`에서는 `actor.chease.gfile`을 free-boundary 변환하고 다시 `gEQDSK2IMAS(...)`로 넘긴다.

위험 지점:

- `actor.chease.gfile.psirz .= psi_free_rz'`에서 shape가 맞지 않으면 array error가 날 수 있다.
- `gEQDSK2IMAS(...)`가 `g.psi`, `g.pres`, `g.pprime`, `g.fpol`, `g.ffprim`, `g.qpsi` 길이 일관성을 확인하지 않고 그대로 넣는다.
- `eqt.boundary.strike_point` 또는 `eqt.boundary.x_point` 길이가 0인 상태에서 weight 계산이 division by zero 또는 empty control point 문제를 만들 수 있다.
- `rho_pol = sqrt.(psin)`에 duplicate, NaN, non-monotonic 값이 있으면 CHEASE input `EXPEQ`는 형식상 써지지만 solver 내부에서 나중에 array error를 낼 수 있다.

### 3.3 CHEASE.jl package 쪽 risk point

`CHEASE.run_chease(...)` 자체는 매번 `mktempdir()`를 쓰므로 같은 directory를 재사용하는 구조는 아니다. 따라서 단순 workdir 충돌 가능성은 낮다.

하지만 다음 리스크는 남는다.

- `write_EXPEQ_file(...)`는 length equality만 확인한다.
- `rho_pol`이 strict increasing인지 확인하지 않는다.
- `rho_pol[1] == 0`, `rho_pol[end] == 1`을 확인하지 않는다.
- `pressure`, `j_tor` finite check가 없다.
- `mode=82`를 쓰지만 `rho_pol`, `pressure`, `j_tor` sampling density가 CHEASE 기대와 맞는지 actor에서 보장하지 않는다.
- CHEASE executable 실패 시 `chease.output` 마지막 100줄만 보여주며, 입력 `EXPEQ`와 namelist를 보존하지 않는다. `clear_workdir=true`이면 정상 종료 후 directory는 삭제된다.

### 3.4 root-cause 분리 전략

반복 CHEASE 에러는 다음 네 가지로 분리한다.

1. FUSE `prepare`가 만든 1D input array가 이미 깨져 있는 경우.
2. CHEASE executable이 특정 profile shape에서 fail하는 경우.
3. CHEASE output gfile은 생성되었지만 `MXHEquilibrium.efit` 또는 `gEQDSK2IMAS` 변환에서 fail하는 경우.
4. free-boundary 변환 `VacuumFields.fixed2free(...)` 또는 `psirz` shape overwrite에서 fail하는 경우.

이를 위해 `_step`과 `_finalize` 경계마다 error tag를 다르게 남긴다.

예:

- `CHEASE_INPUT_VALIDATION_FAIL`
- `CHEASE_EXECUTION_FAIL`
- `CHEASE_GFILE_READ_FAIL`
- `CHEASE_FIXED2FREE_FAIL`
- `CHEASE_GEQDSK2IMAS_FAIL`

### 3.5 CHEASE 입력 guard 계획

`ActorCHEASE._step`에서 `CHEASE.run_chease(...)` 호출 전 다음 helper를 호출한다.

```julia
rho_pol, pressure, j_tor = sanitize_chease_profiles(eqt1d, Ip; n=181)
```

동작:

- local `psin = collect(eqt1d.psi_norm)` 또는 explicit normalized psi를 만든다.
- `rho_pol = sqrt.(max.(psin, 0.0))`
- finite가 아닌 point 제거 또는 fail.
- `rho_pol` 기준 sort.
- duplicate `rho_pol` 제거.
- endpoint 보정: first=0, last=1.
- `pressure`를 같은 grid에서 interpolation.
- `j_tor`를 같은 grid에서 interpolation.
- CHEASE에 넘기기 전 일정한 `rho_pol` grid로 resample하는 옵션 제공.
- current sign filtering은 기존처럼 유지하되, sign filtering 후 전체 current가 너무 작아지면 fail.

초기 정책:

- `repair=false` default로 diagnostic fail을 명확히 본다.
- 원인 확인 후 `repair=true`를 CHEASE actor parameter로 추가한다.

### 3.6 CHEASE workdir 보존 옵션

현재 `clear_workdir`는 정상 종료 후 temp dir 삭제 여부만 제어한다. 실패 시에는 temp dir이 남을 수 있지만, actor log와 연결이 약하다.

추가 parameter 후보:

```julia
debug_keep_workdir::Bool = false
debug_copy_inputs_to::String = ""
```

실패 시 보존할 파일:

- `EXPEQ`
- `chease_namelist`
- `chease.output`
- `EQDSK_COCOS_01.OUT`가 있으면 output
- FUSE에서 넘긴 `rho_pol`, `pressure`, `j_tor`, `boundary.r`, `boundary.z` CSV

이렇게 해야 array error가 CHEASE executable 내부인지, FUSE output conversion인지 정확히 볼 수 있다.

### 3.7 free-boundary 변환 격리

CHEASE 자체 fixed-boundary solve와 FUSE free-boundary 변환을 분리한다.

테스트 matrix:

1. `act.ActorEquilibrium.model = :CHEASE`, `act.ActorCHEASE.free_boundary = false`
2. 같은 case에서 `act.ActorCHEASE.free_boundary = true`
3. `gEQDSK2IMAS`만 독립 호출
4. `VacuumFields.fixed2free(...)`만 독립 호출

만약 `free_boundary=false`에서는 반복 안정하고 `true`에서만 깨지면 원인은 CHEASE executable보다 `fixed2free` 또는 `psirz` shape overwrite 쪽이다.

### 3.8 CHEASE 반복 regression test

테스트 후보:

```julia
for k in 1:10
    actor = ActorEquilibrium(dd, act; model=:CHEASE)
    # 또는 prepare -> step(CHEASE) -> finalize(CHEASE)
end
```

단, 매 loop에서 같은 `dd`를 그대로 쓸 때와 `deepcopy(dd0)`를 쓸 때를 나눈다.

판정:

- `deepcopy(dd0)` 반복은 되는데 같은 `dd` 누적 반복만 깨지면 FUSE actor sequencing/state mutation 문제다.
- 둘 다 같은 profile에서 깨지면 CHEASE input/profile 문제다.
- 특정 generation 이후에만 깨지면 optimizer가 만든 profile bound/repair 문제다.

### 3.9 CHEASE 수정 순서

1. `_step` 전 input snapshot 저장.
2. `sanitize_chease_profiles` 없이 validation만 먼저 추가.
3. 반복 실패 case에서 fail tag와 input CSV 확보.
4. `free_boundary=false` A/B test로 failure 위치 분리.
5. input이 문제이면 `sanitize_chease_profiles` repair 옵션 구현.
6. `fixed2free`가 문제이면 `psi_free_rz` shape, transpose, `EQ.r`, `EQ.z`, `gfile.psirz` dimension assert 추가.
7. `gEQDSK2IMAS`가 문제이면 GEQDSK 1D array lengths assert 추가.
8. 정상화 후 repeated-CHEASE regression test를 CI 또는 local test로 고정.

---

## 4. Bayesian optimization 기능 설계

### 4.1 요구사항

- 기존 genetic algorithm workflow와 비슷하게 `ini`의 `OptParameter` bounds를 사용한다.
- FUSE run 하나가 비싸므로 sample efficiency가 좋아야 한다.
- HPC worker 병렬 평가를 지원해야 한다.
- 실패 case는 objective penalty로 처리하고, 저장 schema는 기존 study database 흐름과 최대한 맞춘다.
- 의존성 추가는 최소화한다.
- BayesianOptimization.jl 전체를 dependency로 넣기보다는 필요한 기능만 작게 넣는다.

### 4.2 외부 BayesianOptimization.jl을 그대로 쓰지 않는 이유

장점:

- 이미 `BOpt`, `ExpectedImprovement`, `UpperConfidenceBound`, Sobol/LHS initializer, GP wrapper가 있다.

단점:

- `GaussianProcesses`, `NLopt`, `Sobol`, `ElasticArrays`, `ElasticPDMats`, `DiffResults` 등 의존성이 추가된다.
- HPC worker에서 precompile 비용과 package compatibility 리스크가 커진다.
- FUSE의 기존 `optimization_engine(...)`, HDF5/CSV 저장, restart worker policy와 직접 맞물리지 않는다.
- FUSE objective는 multi-objective/constraint 구조인데, 일반 BO는 scalar objective 중심이라 adapter가 필요하다.

결론:

- 1차 구현은 FUSE 내부 경량 Bayesian optimizer로 간다.
- 나중에 필요하면 `BayesianOptimization.jl`를 weak dependency/extension으로 붙인다.

### 4.3 FUSE 내부 경량 BO 구조

새 파일 후보:

- `src/workflows/bayesian_optimization_workflow.jl`
- `src/studies/bayesian_optimization.jl`
- `src/bayesian_optimization.jl`

`src/FUSE.jl` include 순서:

- `include("optimization.jl")` 뒤에 `include("bayesian_optimization.jl")`
- studies include 구간에 `include(joinpath("studies", "bayesian_optimization.jl"))`
- workflows include 구간에 `include(joinpath("workflows", "bayesian_optimization_workflow.jl"))`

### 4.4 API 초안

Study type:

```julia
StudyBayesianOptimizer(
    sty::ParametersStudy,
    ini::ParametersAllInits,
    act::ParametersAllActors,
    constraint_functions::Vector{IMAS.ConstraintFunction},
    objective_function::IMAS.ObjectiveFunction;
    kw...
)
```

Workflow:

```julia
workflow_bayesian_optimization(
    ini,
    act,
    actor_or_workflow,
    objective_function,
    constraint_functions;
    initial_samples,
    iterations,
    batch_size,
    acquisition,
    save_folder,
    save_dd,
    database_policy,
    kw...
)
```

State:

```julia
mutable struct BayesianOptimizationState
    X::Matrix{Float64}
    y::Vector{Float64}
    constraints::Matrix{Float64}
    status::Vector{Symbol}
    bounds::Matrix{Float64}
    opt_parameter_names::Vector{String}
    best_x::Vector{Float64}
    best_y::Float64
    iteration::Int
    rng_seed::Int
    acquisition::Symbol
end
```

### 4.5 Objective scalarization

초기 BO는 scalar objective만 공식 지원한다.

기존 GA의 multi-objective와 호환하려면 두 가지 모드를 둔다.

Mode 1: 단일 objective

- `objective_function::IMAS.ObjectiveFunction`
- `constraint_functions`는 penalty로 처리한다.
- feasible이면 `score = objective`.
- infeasible이면 `score = objective + penalty_scale * constraint_violation`.
- FUSE objective가 minimize convention이면 그대로 minimize한다.

Mode 2: scalar merit function

- 사용자가 직접 `merit(dd)::Float64`를 넘긴다.
- 예: KDEMO에서는 `Pnet` maximize, `Pfusion ~ 1500 MW`, `fGW <= 1`, `q` monotonic, `Pdiv/PLH` 제한 등을 하나의 penalty score로 만든다.

권장:

- FUSE 내부 BO core는 minimize 기준으로 통일한다.
- maximize metric은 adapter에서 `-metric`으로 변환한다.

### 4.6 초기 sampling

BayesianOptimization.jl은 `ScaledSobolIterator`와 `ScaledLHSIterator`를 제공한다.

FUSE 내부 1차 구현:

- Sobol dependency를 피하기 위해 Latin Hypercube Sampling을 직접 구현한다.
- 필요한 코드는 작다.
- 각 dimension에서 `[lower, upper]`를 `N`개 bin으로 나누고 random permutation한다.

초기 sample 수:

- `initial_samples = max(2D + 1, batch_size)` 권장.
- FUSE expensive objective에서는 `5D` 이상도 고려.
- KDEMO처럼 variable 수가 적은 경우 `initial_samples = 2D~5D`가 적당하다.

### 4.7 Surrogate model

1차 구현은 squared-exponential Gaussian process를 직접 구현한다.

필요 dependency:

- `LinearAlgebra`
- `Random`
- `Statistics`
- `SpecialFunctions`는 이미 FUSE dependency에 있으므로 EI의 normal CDF/PDF에 사용 가능.

Model:

```julia
k(x, x') = σf^2 * exp(-0.5 * sum(((x - x') ./ ℓ).^2)) + σn^2 * I
```

입력 scaling:

- `x_scaled = (x - lower) ./ (upper - lower)`

출력 scaling:

- `y_scaled = (y - mean(y)) / std(y)`

Hyperparameters:

- 초기 구현은 full optimization을 피한다.
- `ℓ = 0.2~0.5` 범위 기본값.
- `σf = 1`
- `σn = 1e-6` 또는 noisy FUSE objective라면 `1e-4`.

2차 구현:

- `Optim.jl`은 이미 FUSE dependency이므로 negative log marginal likelihood를 `Optim.optimize`로 fit할 수 있다.
- 단, full BO loop 안정화 전에는 hyperparameter optimization을 끄는 옵션을 둔다.

### 4.8 Acquisition functions

필수:

- `ExpectedImprovement`
- `UpperConfidenceBound`

EI:

```julia
z = (best_y - mu) / sigma
EI = (best_y - mu) * Phi(z) + sigma * phi(z)
```

minimize 기준이다.

UCB/LCB:

```julia
LCB = mu - kappa * sigma
```

minimize 기준에서는 LCB가 작은 점을 고른다.

### 4.9 Acquisition optimization

NLopt dependency를 피한다.

1차 구현:

- candidate pool random/LHS `n_candidates = 1000~10000`
- surrogate/acquisition을 pool에서 평가
- top candidates 중 기존 sample과 너무 가까운 점 제거
- batch size만큼 고른다.

2차 구현:

- `Optim.jl` random restart local optimize를 추가한다.
- 단, box-bound 처리와 gradient 안정화가 필요하므로 1차에서는 pool 방식이 더 안전하다.

### 4.10 Batch BO / HPC worker 연동

FUSE는 case 하나가 비싸고 `n_workers=55` 같은 병렬 평가가 필요하다. 순수 sequential BO는 비효율적이다.

Batch proposal:

1. acquisition 상위 candidate pool을 만든다.
2. 가장 좋은 candidate를 선택한다.
3. 선택된 candidate 주변 acquisition을 penalize한다.
4. 최소 distance 기준으로 batch diversity를 보장한다.
5. `batch_size`개를 기존 `optimization_engine(..., X::AbstractMatrix, ...)` 형태로 병렬 평가한다.

초기 설정:

- `batch_size = min(n_workers, requested_batch_size)`
- batch BO 한 iteration = `batch_size` FUSE evaluations
- `iterations`는 BO update 횟수로 정의한다.
- total evaluations = `initial_samples + iterations * batch_size`

### 4.11 기존 저장/실패 처리 재사용

`optimization_engine(...)`는 이미 다음을 처리한다.

- worker별 deepcopy
- `parameters_from_opt!(ini, x)`
- run directory/HDF5 저장
- CSV extract 저장
- 실패 시 `ff=Inf`, `gg=Inf`
- `GC.gc()`, `malloc_trim_if_glibc()`

따라서 BO도 새 evaluator를 만들지 않고 기존 `_optimization_engine` 또는 matrix version `optimization_engine(...)`를 호출한다.

추가 저장:

- `bayesian_state.jls`
- `bayesian_observations.csv`
- `bayesian_candidates_genXXXX.csv`
- `bayesian_surrogate_summary.json`

`extract.csv`에는 기존 column 외에 아래 column을 추가할 수 있다.

- `bo_iter`
- `bo_case`
- `bo_score`
- `bo_acquisition`
- `bo_feasible`

### 4.12 restart/checkpoint 정책

HPC에서 BO는 중간 실패 가능성이 높다.

정책:

- 매 BO iteration 종료 후 state serialize.
- `tmp_csv_output` merge 후 observations를 다시 읽어 state 재구성 가능하게 한다.
- `continue_state` 또는 `load_bayesian_optimization(path)` 제공.
- worker restart는 `StudyMultiObjectiveOptimizer.restart_workers_after_n_generations`와 동일한 패턴을 재사용한다.

### 4.13 BO와 GA의 관계

초기 대규모 탐색은 GA가 유리하다.

BO 사용 위치:

- GA smoke/pilot 후 feasible region이 어느 정도 잡힌 뒤 local refinement.
- KDEMO case 2처럼 R0/B0/Ip/aspect fixed이고 `ne`, `Te`, heating split, deposition 위치 등 변수 수가 제한된 상황.
- 실패율이 높을 때는 BO surrogate가 failure boundary를 피하는 데 도움이 된다.

권장 sequence:

1. GA 또는 random/LHS로 initial feasible database 확보.
2. feasible + high-score points를 BO initial observations로 import.
3. BO batch refinement.
4. 필요하면 다시 GA population seed로 best BO points를 넣는다.

---

## 5. 구현 phase 계획

### Phase 0: 재현 case 고정

목표:

- 현재 관측된 stepped pressure와 CHEASE array error를 재현 가능한 script로 고정한다.

작업:

1. FUSE clone에서 user KDEMO smoke input 또는 가장 작은 재현 input을 하나 만든다.
2. TEQUILA/CHEASE 각각 `ActorEquilibrium`만 돌리는 최소 script를 만든다.
3. `prepare` 전후와 solver finalize 후 profile CSV를 남긴다.
4. 실패 CHEASE case의 `EXPEQ`, `chease_namelist`, `chease.output`을 보존한다.

산출물:

- `test/equilibrium/reproduce_core_pprime_flattening.jl`
- `test/equilibrium/reproduce_chease_repeated_array_error.jl`
- `scratch/equilibrium_debug/*.csv` 또는 test artifact

### Phase 1: profile preparation fix

목표:

- solver에 들어가기 전 pressure/current profile이 axis 부근에서 불필요하게 flat해지는 문제를 막는다.

작업:

1. `ActorEquilibrium.prepare`에 explicit normalized psi local variable을 도입한다.
2. `psi_norm > 0.02` hard cutoff를 제거하거나 parameter화한다.
3. axis regularity는 ghost point 또는 low-order even fit으로 처리한다.
4. `validate_equilibrium_1d_input!`를 추가한다.
5. TEQUILA/CHEASE 양쪽에서 같은 input validator를 사용한다.

검증:

- analytic pressure regression.
- 기존 ITER/STEP/baby-MANTA case smoke.
- KDEMO case 2 mini workflow.

### Phase 2: CHEASE repeated-run hardening

목표:

- 반복 CHEASE 실행에서 array error 원인을 분리하고, FUSE actor 쪽에서 방지 가능한 입력/shape 문제를 막는다.

작업:

1. `sanitize_chease_profiles` 또는 `validate_chease_profiles` 추가.
2. CHEASE `_step` 전 input diagnostics 추가.
3. `gEQDSK2IMAS` 전 GEQDSK array length/finite assert 추가.
4. `fixed2free` 전후 `psirz` shape assert 추가.
5. `free_boundary=false/true` repeated test 분리.
6. 실패 시 CHEASE input/output artifact 보존 옵션 추가.

검증:

- same `dd` repeated CHEASE 10회.
- `deepcopy(dd0)` repeated CHEASE 10회.
- `free_boundary=false`와 `true` 각각 10회.
- StationaryPlasma loop 내부 CHEASE 반복 3-5회.

### Phase 3: Bayesian optimization core

목표:

- 외부 heavyweight dependency 없이 BO core를 FUSE 내부에 추가한다.

작업:

1. `BayesianOptimizationState` 정의.
2. LHS initializer 구현.
3. lightweight GP fit/predict 구현.
4. EI/LCB acquisition 구현.
5. candidate-pool acquisition optimizer 구현.
6. batch proposal 구현.
7. 기존 `optimization_engine`과 연결한다.

검증:

- Branin 함수 unit test.
- noisy quadratic unit test.
- failed objective penalty test.
- FUSE mock actor/workflow test.

### Phase 4: Study/Workflow integration

목표:

- 기존 `StudyMultiObjectiveOptimizer`와 같은 사용감으로 BO를 실행한다.

작업:

1. `StudyBayesianOptimizer` 추가.
2. `study_parameters(::Val{:BayesianOptimizer})` 추가.
3. `workflow_bayesian_optimization(...)` 추가.
4. `save_bayesian_optimization`, `load_bayesian_optimization` 추가.
5. `database_policy=:single_hdf5` 지원.
6. worker restart 정책 추가.

검증:

- local `n_workers=0` smoke.
- local distributed small `n_workers=2` smoke.
- HPC smoke는 production FUSE와 분리된 clone에서 별도 진행.

### Phase 5: KDEMO optimizer 적용

목표:

- user KDEMO case 2 중심으로 GA 결과를 BO에 연결한다.

작업:

1. case 2 고정값: R0, B0, aspect ratio, Ip 고정.
2. variable: density/temperature shape, heating split, deposition spread, HCD radial locations, allowed current drive constraints.
3. constraints: q monotonic/no reversal, fGW <= 1, Pfusion near 1500 MW, Pdiv/PLH 제한, betaN 목표, net power maximize.
4. GA feasible database를 BO initial observations로 import.
5. BO batch refinement queue를 구성한다.

주의:

- 이 phase는 FUSE source 기능이 안정화된 뒤 production queue와 연결한다.
- source patch 검증 전에는 현재 돌아가는 queue의 environment를 바꾸지 않는다.

---

## 6. Acceptance criteria

### 6.1 `P'` flattening fix acceptance

통과 조건:

- analytic pressure input에서 `prepare` 직후 first non-axis points의 finite-difference `dP/dpsi`가 모두 near-zero로 붕괴하지 않는다.
- TEQUILA finalize output에서 first 5 core points 전체가 near-zero `dpressure_dpsi`가 되는 현상이 사라진다.
- CHEASE output `g.pprime` 기반 `dpressure_dpsi`도 같은 기준을 만족한다.
- 기존 equilibrium smoke cases에서 q, pressure, Ip가 크게 regression하지 않는다.

정량 기준 초안:

- `count(abs.(dP_dpsi[2:6]) .< 1e-3 * maximum(abs.(dP_dpsi))) <= 1`
- pressure monotonic violation count가 0 또는 매우 작은 tolerance 내.
- `norm(P_prepare - P_core_profiles) / norm(P_core_profiles) < 1e-2` for analytic test.

### 6.2 CHEASE repeated-run acceptance

통과 조건:

- 같은 input `dd`를 `deepcopy`해서 CHEASE 10회 반복 성공.
- 같은 `dd` 객체에 대해 `ActorEquilibrium(model=:CHEASE)` 반복 5회 이상 성공.
- `free_boundary=false`와 `true` 모두에서 array length/shape assert 통과.
- 실패 시에는 error tag와 input artifact가 남아 원인 추적이 가능하다.

### 6.3 Bayesian optimization acceptance

통과 조건:

- Branin-like benchmark에서 초기 LHS 대비 best score가 개선된다.
- failed objective가 섞여도 state 저장/재시작이 가능하다.
- `n_workers > 1`에서 batch evaluation이 기존 GA처럼 HDF5/CSV output을 남긴다.
- `OptParameter exceed bounds` 같은 candidate bounds 문제가 발생하지 않도록 모든 BO proposal이 `float_bounds(opt_ini)` 내부에 clamp된다.
- BO state에서 best candidate와 corresponding run directory를 추적할 수 있다.

---

## 7. 우선순위와 예상 patch 순서

가장 먼저 할 일:

1. `ActorEquilibrium.prepare` diagnostic을 넣고 stepped pressure가 solver 전부터 생기는지 확인한다.
2. CHEASE input/output validator를 넣어 반복 에러가 input array 문제인지 fixed2free/gEQDSK 문제인지 분리한다.
3. `psi_norm > 0.02` hard cutoff를 제거/완화하는 patch를 만든다.
4. TEQUILA/CHEASE regression test를 돌린다.
5. 그 다음 BO core를 기존 `optimization_engine` 재사용 방식으로 붙인다.

이 순서가 맞는 이유:

- `P'` 문제는 BO/GA 결과 품질을 직접 망가뜨리므로 optimizer 추가보다 먼저 잡아야 한다.
- CHEASE 반복 에러는 optimizer에서 실패율을 키우므로 BO를 붙이기 전에 최소한 failure classification과 retry 가능성을 확보해야 한다.
- BO는 existing GA workflow 위에 얹으면 되지만, equilibrium output이 신뢰할 수 없으면 surrogate가 잘못된 target을 학습한다.

---

## 8. 열린 확인 사항

확인이 필요한 점:

- IMAS `eqt1d.psi_norm`을 직접 set하는 것이 schema/expression 정책상 안전한지 확인해야 한다.
- `pressure`를 `psi_norm` 기준으로 보존할지, `rho_pol` 기준 axis regularity를 우선할지 결정해야 한다.
- CHEASE `mode=82`가 현재 FUSE input profile 방식에 최적인지 확인해야 한다.
- CHEASE.jl package 자체에 validator patch를 넣을지, FUSE actor에만 guard를 둘지 결정해야 한다.
- BO objective convention을 FUSE의 `ObjectiveFunction` minimize convention으로 완전히 통일할지, maximize adapter를 user-facing API에 노출할지 결정해야 한다.

---

## 9. 결론

현재 코드 기준으로 core `P'` flattening의 1차 원인은 CHEASE/TEQUILA solver 내부보다 `ActorEquilibrium.prepare`의 공통 preprocessing일 가능성이 높다. 특히 `psi_norm > 0.02` cutoff 후 mirror-PCHIP로 axis zero-gradient를 강제하는 방식이 first several core points를 flat하게 만들 수 있다.

CHEASE 반복 array 에러는 `run_chease`가 temp directory를 매번 새로 쓰므로 단순 workdir 재사용 문제 가능성은 낮고, FUSE actor가 넘기는 `rho_pol/pressure/j_tor` 배열 조건 또는 free-boundary/gEQDSK 변환 shape 문제가 더 유력하다.

Bayesian optimization은 `BayesianOptimization.jl` 전체 의존성을 바로 추가하지 말고, 기존 FUSE `optimization_engine`과 study database 저장 체계를 재사용하는 내부 경량 BO부터 구현하는 것이 가장 안전하다.
