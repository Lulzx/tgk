TGK - Tiny GyroKinetic code
===========================

TGK is a local delta-f flux-tube gyrokinetic turbulence code in about 4000
lines of C99. It needs only libc, libm and pthreads, and runs on a single
machine. It is tuned for Apple Silicon (NEON with ARMv8.3 complex
instructions) but builds with any C99 compiler.

Features
--------

- Electrostatic ITG and TEM turbulence with adiabatic or kinetic electrons,
  validated against GS2 and GX (see `validation/`). Electromagnetic (A_par)
  terms are implemented but NOT validated: kinetic-electron EM runs have a
  known spurious instability.
- Pseudo-spectral (kx, ky) with 2/3-rule dealiasing; custom mixed-radix
  (2, 3, 4, 5) Stockham FFTs.
- 4th order finite differences in z and v_par, twist-and-shift boundary
  condition, Gauss-Laguerre quadrature in mu.
- s-alpha and Miller local equilibrium geometry (Bishop's method; matches
  GS2's `eik` to within 1%).
- RK4 with adaptive CFL time step. The optional implicit (IMEX) parallel
  streaming for kinetic electrons is known to be wrong; do not use it.
- Linearized Dougherty collision operator with discrete conservation of
  density, momentum and energy.
- FP32 state with FP64 field reductions, or full FP64 (`tgk_fp64`).
- Free-energy balance diagnostic (drive, hyperdiffusion, residual).
- Bit-identical checkpoint restart.

Building
--------

    make            # builds tgk and tgk_fp64
    make check      # validation suite in both precisions (a few minutes)

Usage
-----

    ./tgk tests/cyclone_nonlinear.in
    ./tgk tests/kinetic_em.in        # EM kinetic electrons: unvalidated, see below

Input files are plain `key = value` lists; see the files in `tests/`.
Output is a text trace (`<prefix>.trace.txt`) with time, |phi|^2, fluxes,
diffusivities, growth rate, frequency and the free-energy budget, plus
optional binary snapshots and checkpoints.

Normalization
-------------

Lengths in units of the minor radius a, velocities in v_t = sqrt(T/m),
perpendicular wavenumbers in 1/rho. The gradients `fprim` and `tprim` are
R/L_n and R/L_T; `rmaj` is R/a.

Validation
----------

`validation/README.md` has the full comparison against GS2 and GX on the
Cyclone Base Case, with tables, plots and scripts. In summary:

- Linear s-alpha CBC (adiabatic electrons): at kyρ ≤ 0.45, γ agrees with GS2
  and GX to within 2–4% and ω to within about 1%. At kyρ = 0.3, TGK gives
  γ = 0.0936 and ω = 0.282; GX gives 0.0930 and 0.282.
- Miller geometry: metric and drift coefficients agree with GS2's `eik`
  to within 1%. Elongation is stabilizing, as in GS2: peak γ is 0.141,
  0.116 and 0.087 at κ = 1.0, 1.4 and 1.8 (GS2: 0.143, 0.118 and 0.088).
- Kinetic electrons, electrostatic: γ agrees with GX to within 3–4% at
  kyρ = 0.2–0.4.
- Nonlinear CBC (s-alpha, GX parameters, ν = 0.002): χ_i ≈ 1.0–1.2 ρ² v_t/a
  on grids from 32 × 16 to 128 × 32. The differences between grids are
  within the statistical error.
- Not validated: electromagnetic runs and the IMEX path (see above).

`make check` tests:

- FFT round trip, Parseval, and exact conservation of the dealiased E x B
  bracket.
- RK4 convergence order.
- Linear CBC at kyρ = 0.3 against GX: s-alpha, circular Miller, and
  electrostatic kinetic electrons. Also checks that elongation is
  stabilizing and that a sub-critical case decays.
- Free-energy balance closes: to about 1% in the linear phase and 2% on
  average in saturated turbulence.
- Rosenbluth-Hinton zonal flow residual within 30% of the analytic value.
- Nonlinear CBC saturation on a small grid (sanity bounds only; see
  `validation/` for converged values).
- Collisional conservation, checkpoint restart.
- FP32 vs FP64: pointwise agreement in the linear phase, statistical
  agreement in saturated turbulence.

Performance
-----------

Cyclone resolution (128 x 32 x 24 x 32 x 8, 25M points), Apple M4 Pro,
8 performance cores: 76 ms per RK stage in FP32, 140 ms in FP64.

The fused stage kernel reaches about 13 GB/s, far below the memory
bandwidth of the machine; it is limited by compute and FFT latency, not
by memory traffic.

Discretization notes
--------------------

Parallel streaming and the mirror force are discretized on h/sqrt(F0), the
variable in which the free energy is an unweighted norm. Their compressional
terms then cancel exactly and the remaining centered stencils cannot create
energy, including near v_max where 1/F0 is large. The even-nkx Nyquist kx
column is excluded from the E x B bracket, which conserves energy exactly.

Files
-----

    main.c    input parser, thread pool, RK4 driver
    rhs.c     fused right-hand side: streaming, mirror, drifts, drive, E x B
    fft.c     2D dealiased real FFTs
    grid.c    wavenumber and velocity grids, quadrature, Bessel functions
    field.c   quasineutrality and Ampere's law
    imex.c    implicit kinetic-electron streaming
    coll.c    Dougherty collision operator
    geom.c    s-alpha and Miller geometry
    diag.c    fluxes, growth rates, free energy, snapshots, checkpoints
    simd.h    NEON complex arithmetic

License
-------

MIT. See LICENSE.
