# TGK — Tiny Gyrokinetic Turbulence Code

**TGK** ("tiny gyrokinetic") is a single-node, zero-dependency, local $\delta f$ flux-tube gyrokinetic turbulence code written in ~3,600 lines of C99 (`libc`, `libm`, `pthreads`) and tuned for **Apple Silicon (M4 Pro)**.

Designed in the spirit of Fabrice Bellard's projects, TGK makes standard nonlinear tokamak core turbulence simulations (ITG, TEM, and electromagnetic KBM regimes) something one person can run on a single workstation in an afternoon—without external libraries, MPI, or supercomputer allocations.

---

## Design Principles

1. **Small & Self-Contained**: The entire solver—including its mixed-radix 2D real-to-complex FFT with folded 2/3-rule dealiasing, Golub–Welsch Gauss–Laguerre quadrature, Miller equilibrium geometry, implicit kinetic-electron solver, and conserving Dougherty collision operator—fits in **3,608 lines of C99** across 10 core source files.
2. **Bandwidth-Aware Fused Architecture**: Gyrokinetic time integration is memory-bandwidth bound. TGK stores the 5D distribution function in contiguous `[species][μ][v∥][z][ky][kx]` slabs sized to fit in L2 cache. Each RK4 stage executes in a **single fused pass** per slab: computing parallel streaming, mirror force, curvature/$\nabla B$ drifts, equilibrium gradient drive, pseudo-spectral $E \times B$ advection, the in-place RK stage update, and the next stage's FP64 velocity-space field integrals simultaneously.
3. **Apple Silicon Native**:
   - **ARM64 NEON + ARMv8.3-A Complex Intrinsics (`FEAT_FCMA`)**: Uses hardware `vcmlaq_f32` and `vcmlaq_rot90_f32` instructions in `simd.h`.
   - **128-Byte Cache-Line Alignment**: All 5D state arrays, geometry tables, and per-thread workspaces are aligned to the M4 Pro's 128-byte cache lines.
   - **P-Core QoS Thread Pool**: Automatically queries `hw.perflevel0.physicalcpu` via `sysctlbyname` and pins worker threads to Performance cores using `QOS_CLASS_USER_INTERACTIVE`.
4. **Checkable Physics**: Ships with an automated verification suite (`make check`) testing mathematical primitives, linear ITG/TEM/KBM dispersion, Rosenbluth–Hinton residual zonal flows, nonlinear Cyclone Base Case saturation & Dimits shift, exact discrete collisional conservation, bit-identical checkpoint restarts, and mixed-precision (`FP32` state + `FP64` pairwise reductions) vs. full `FP64` agreement to within $< 0.5\%$.

---

## Physics & Numerical Model

TGK evolves the non-adiabatic part of the perturbed gyrocenter distribution function $h_s(k_x, k_y, z, v_\parallel, \mu, t)$ in field-aligned flux-tube coordinates:

$$\frac{\partial h_s}{\partial t} + v_{\text{th},s} v_\parallel (\hat{\mathbf{b}}\cdot\nabla z) \frac{\partial g_s}{\partial z} - v_{\text{th},s} \mu (\hat{\mathbf{b}}\cdot\nabla B) \frac{\partial h_s}{\partial v_\parallel} + i \omega_{ds} g_s + \{\langle \chi_s \rangle_{\mathbf{R}}, h_s\} = i \omega_{*s}^T F_{0s} \langle \chi_s \rangle_{\mathbf{R}} + C[h_s]$$

where $\langle \chi_s \rangle_{\mathbf{R}} = J_0(k_\perp \rho_s) (\phi - v_{\text{th},s} v_\parallel A_\parallel)$ is the gyroaveraged generalized potential and $g_s = h_s + \frac{Z_s}{T_s} F_{0s} \langle \chi_s \rangle_{\mathbf{R}}$.

| Component | Implementation |
|---|---|
| **Perpendicular $(k_x, k_y)$** | Spectral representation; pseudo-spectral $E \times B$ Poisson bracket $\{\chi, h\}$ via custom Stockham auto-sorting radix-2/3/5 2D FFTs with folded 2/3-rule dealiasing and paired C2R transforms |
| **Parallel $(z)$** | 4th-order centered finite differences + 4th-order parallel hyperdiffusion; twist-and-shift boundary condition linking $k_x$ chains across $z = \pm \pi N$ |
| **Velocity Space $(v_\parallel, \mu)$** | 4th-order centered differences in $v_\parallel \in [-v_{\max}, v_{\max}]$; Golub–Welsch Gauss–Laguerre quadrature in $\mu$ (exact for polynomials $\times$ Maxwellian) |
| **Field Equations** | Algebraic gyrokinetic quasineutrality (with exact discrete $\Gamma_0(b_s) = \int d^3v\, F_{0s} J_0^2$ and flux-surface-averaged zonal flow response $\langle\phi\rangle_{\text{FS}}$ at $k_y = 0$) + parallel Ampère's law for $A_\parallel$ |
| **Equilibrium Geometry** | Analytic $s$-$\alpha$ model and shaped **Miller local equilibrium** ($\kappa, s_\kappa, \delta, s_\delta, \Delta'$) |
| **Time Integration** | Explicit 4th-order Runge–Kutta (RK4) with adaptive linear + nonlinear CFL timestep, plus optional **IMEX** implicit kinetic-electron parallel streaming along linked ballooning chains |
| **Collisions** | Linearized Dougherty (Lenard–Bernstein) operator with implicit $v_\parallel$ drag/diffusion, $\mu$ pitch-angle scattering, and exact discrete projection conserving $\delta n$, $\delta u_\parallel$, and $\delta E$ to machine precision |

---

## Quick Start

### 1. Build & Run Validation Suite

Requires only a C99 compiler (`clang` or `gcc`), `libm`, and `pthreads`:

```sh
make check
```

This builds both the default mixed-precision binary (`./tgk`) and the full double-precision binary (`./tgk_fp64`), executes all 6 verification suites in both precisions, and verifies that mixed precision matches full FP64 within $< 0.5\%$.

### 2. Run a Simulation

```sh
# Nonlinear Cyclone Base Case (adiabatic electrons)
./tgk tests/cyclone_nonlinear.in

# Shaped Miller geometry with kinetic electrons, electromagnetic A_par, and IMEX
./tgk tests/kinetic_em.in
```

---

## Validation & Benchmark Results (Apple M4 Pro)

Output from `make check` on an **Apple M4 Pro** (8 Performance cores + 4 Efficiency cores):

```text
--- Test 1: Gauss-Laguerre Quadrature & 2D Dealiased FFT (Radix 2, 3, 5) ---
  [PASS] Gauss-Laguerre moments (1, x, x^2, x^3): 1.000000000000, 1.000000000000, 2.000000000000, 6.000000000000
  [PASS] 2D Dealiased FFT round-trip max error = 6.743e-07 (pad 48x48) [1.776e-15 in FP64]
  [PASS] 2D FFT Parseval energy conservation: E_spec=1705.532403, E_real=1705.532305

--- Test 2: Linear Cyclone Base Case, TEM, KBM & Miller Geometry ---
  [PASS] Cyclone Base Case (ky=0.3) linear ITG growth rate gamma=0.3367, omega=-0.2642
  [PASS] Cyclone Base Case (ky=0.5) linear ITG growth rate gamma=0.3802 > gamma(0.3), omega=-0.5830
  [PASS] Sub-critical gradient (R/L_T=1.0 < R/L_T_crit) is linearly stable: gamma=-0.1270
  [PASS] Miller shaped geometry (kappa=1.4, delta=0.2) ITG stabilization: gamma_miller=0.2793 < gamma_s_alpha=0.3367
  [PASS] Kinetic electrons + electromagnetic A_par (beta=0.5%) + IMEX growth rate: gamma=0.6429, omega=1.6514

--- Test 3: Rosenbluth-Hinton Residual Zonal Flow Benchmark ---
  [PASS] Rosenbluth-Hinton zonal flow residual |phi(T)|/|phi(0)| = 0.4664 (sustained non-zero)

--- Test 4: Nonlinear Cyclone Base Case Saturation & Dimits Shift ---
  [PASS] Nonlinear Cyclone transport chi_i(R/L_T=6.92)=7.2248e-04 > Dimits-shift regime chi_i(R/L_T=4.8)=3.8992e-04

--- Test 5: Dougherty Collisional Conservation & Bit-Identical Checkpoint Restart ---
  [PASS] Dougherty collision operator conserves density (dn=8.01e-14), momentum (du=7.58e-14), energy (dE=3.17e-14)
  [PASS] Checkpoint restart is 100% bit-identical across 55296 complex state points

--- Test 6: Apple M4 Pro Fused RK Stage Bandwidth Benchmark ---
  [PASS] M4 Pro (8 P-cores, FP32): 76.56 ms/stage (306.22 ms/step) on 25.17M grid points (192.0 MB/copy, 13.1 GB/s fused bandwidth)
  [PASS] M4 Pro (8 P-cores, FP64): 148.28 ms/stage (593.10 ms/step) on 25.17M grid points (384.0 MB/copy, 13.6 GB/s fused bandwidth)

=== Comparing FP32-mixed vs Full FP64 Agreement (< 0.5% threshold) ===
  [PASS] FP32-mixed vs Full FP64 relative diff: |phi|^2=0.0000%, Q_i=0.0000%, gamma=0.0001% (all < 0.5%)
```

---

## Codebase Architecture

| File | Lines | Responsibility |
|---|---:|---|
| [`main.c`](main.c) | 616 | Strict `key = value` parser, Apple Silicon P-core thread pool (`QOS_CLASS_USER_INTERACTIVE`), adaptive RK4 loop, CLI |
| [`fft.c`](fft.c) | 505 | Stockham auto-sorting radix-2/3/5 1D & 2D real-to-complex / paired complex-to-real FFTs with folded 2/3-rule dealiasing |
| [`rhs.c`](rhs.c) | 474 | Single-pass fused RK stage kernel (streaming, mirror, drifts, drive, cached-field $E \times B$ bracket, next-stage field reduction) |
| [`grid.c`](grid.c) | 401 | Spectral $(k_x, k_y)$ grids, Golub–Welsch Gauss–Laguerre $\mu$ quadrature, Bessel $J_0$ & $\Gamma_0$, twist-and-shift tables |
| [`diag.c`](diag.c) | 346 | Pairwise FP64 heat/particle fluxes, diffusivity $\chi_s$, growth rates, free-energy balance residual, snapshots & checkpoints |
| [`tgk.h`](tgk.h) | 331 | Simulation data structures, thread workspace definitions, and module interfaces |
| [`imex.c`](imex.c) | 212 | Implicit kinetic-electron parallel streaming along twist-and-shift linked ballooning chains (Thomas & Sherman–Morrison) |
| [`coll.c`](coll.c) | 205 | Linearized Dougherty collision operator with exact discrete conservation of density, momentum, and energy |
| [`field.c`](field.c) | 187 | Thread-local FP64 velocity-space integral reduction, quasineutrality (with zonal-flow $\langle\phi\rangle_{\text{FS}}$), and Ampère solve |
| [`simd.h`](simd.h) | 173 | ARM64 NEON (`vcmlaq_f32`) complex vector abstraction, 128-byte cache-line alignment, recursive pairwise FP64 sum |
| [`geom.c`](geom.c) | 158 | Analytic $s$-$\alpha$ and shaped Miller local equilibrium geometry |
| [`tests/test_suite.c`](tests/test_suite.c) | 492 | Automated validation, conservation, precision comparison, and M4 Pro benchmark suite |
| **Total** | **4,100** | *(3,608 core + 492 test suite)* |

---

## License

Released under the [MIT License](LICENSE).
