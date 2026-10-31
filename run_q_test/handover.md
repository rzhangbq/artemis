# Handover: ADI PML instability and Q tests (run_q_test)

Written Sat 2026-10-31 12:27 PDT. Times below are PDT.

## Immediate state: two tests are running

| test | job / node | case | PML change | restart from | ends at step | expected finish |
|---|---|---|---|---|---|---|
| C | 60616457, nid200444, 4 GPUs | cfl 16, 225 um air | alpha_max = 0.2 and pml_R = 1e-4 | `air225_pmltest_cfl16_alpha0p2/60616457/diags/chk0076000` (6.18 ns) | 92000 (7.48 ns) | ~13:37 |
| D | 60616463, nid200277, 1 GPU | cfl 32, 225 um air | alpha_max = 1.0 (R = 1e-8) | `closerctrlline_cut_pmlall_cfl32_air225/60596002/diags/chk0024000` (3.90 ns) | 30000 (4.88 ns) | ~13:40 |

Both Jupyter sessions end about 14:02. Stopping the Jupyter server kills the run (this already cut one test short).

Output, under `/pscratch/sd/r/rzhangbq/`:
- C: `air225_pmltest_cfl16_alpha0p2_R1e-4/60616457/`
- D: `air225_pmltest_cfl32_alpha1p0/60616463/`

### First thing to do when they finish
Fit the rates, skipping about 0.5 ns after each restart:

    P=/pscratch/sd/r/rzhangbq
    python3 $P/cfl_study/rates.py "C" $P/air225_pmltest_cfl16_alpha0p2_R1e-4/60616457 6.7
    python3 $P/cfl_study/rates.py "D" $P/air225_pmltest_cfl32_alpha1p0/60616463 4.4

`rates.py` prints the frequency of the last component (Ez), which is unreliable when Ez is small. Use Ex for frequency.

Reading the result:
- Rate at or below zero: that setting stops the growth. It is a real radiation loss only if a stronger setting gives the same rate.
- Still small and positive: the PML still adds energy. Next look at PML thickness, `warpx.pml_kappa_max`, and the ADI-PML update itself.

## Main finding: the z-PML makes the resonator grow

Geometry: `inputs_artemis_closerctrlline_cut*`, a 1.65 x 1.1 mm cut with a lumped linear inductor (384.6 pH per edge, 26 edges), dx = dy = 2.5 um, dz = 3 um. Silicon is 225 um thick. Drive is 5.5 GHz (Gaussian, ends about 1.09 ns). The feed empties by about 3.5 ns.

Eobs1 amplitude rate during free ring-down. Nothing in the model dissipates, so all decay goes through the PML.

| z walls | air | cfl | PML | rate (%/ns) | Q | f (GHz) |
|---|---|---|---|---|---|---|
| PEC (x/y PML only) | 75 um | 8 | default | ~0 | >~18000 | 5.663 |
| PML | 75 um | 16 | default | -3.6 | 504 | 5.732 |
| PML | 75 um | 32 | default | -4.9 | 359 | 5.605 |
| PML | 75 um | 32 | R = 1e-4 | -6.2 | 284 | 5.607 |
| PML | 225 um | 16 | default | **+8.87** | -203 | 5.718 |
| PML | 225 um | 32 | default | **+7.39** | -239 | 5.614 |
| PML | 225 um | 16 | alpha = 0.2 | +1.80 | -1000 | 5.721 |
| PML | 225 um | 32 | R = 1e-4 | +1.04 | -1690 | 5.595 |

Default PML: R = 1e-8, kappa_max = 1, alpha_max = 0, m = 3, 10 cells.

Conclusions:
- The 225 um growth is a numerical instability, not energy transfer. It is one exponentially growing mode: the same rate in all components, fixed frequency, and U_E/Ex^2 constant over the domain, so there is no other energy reservoir. The drive is off by then.
- The PML is the source, or at least a major part: changing only PML parameters cut the rate 5-7x. It is not eliminated yet.
- **No all-PML Q is a trustworthy radiation Q yet, including Q = 504 at 75 um.**
- Rates look additive: more air gives about +12.3 %/ns at both cfl values; cfl 16 to 32 gives about -1.4 %/ns at both air thicknesses. This rests on four numbers only.
- **Do not run the 8-qubit chip with the default PML.** It would use the same 225 um of air.

Hypotheses that turned out wrong (do not repeat them):
- "Energy is arriving from elsewhere in the circuit" for the 225 um growth. Ruled out by the energy scaling above.
- "Under-resolved sigma*dt explains the cfl dependence of Q". Matching sigma*dt to cfl 16 moved Q further away (359 to 284).

Other observations:
- cfl also changes probe amplitudes by component at the same time (cfl 32/16: Eobs1 Ex about 1.4x, Eobs1 Ez about 0.3x), so single-cell amplitudes are not converged in cfl.
- ADI frequency error vs cfl was measured on the crosstalk geometry: cfl 16 -0.7 %, cfl 32 -2.7 %, cfl 64 -7.2 %. Plot: `/pscratch/sd/r/rzhangbq/cfl_study/cfl_frequency_shift.png`.
- The cfl-8 all-PML run never ran (job cancelled by request). Deck is `inputs_artemis_closerctrlline_cut_pmlall_cfl8`. Convergence of Q and rate in dt is unverified.
- Chain of 26 junctions: split onset when a junction phase passes about 1.5 rad (from `run_jj_test`).

## Open next steps
1. Read tests C and D.
2. Find a PML setting where the rate stops changing as the PML is strengthened. Only that rate is real radiation loss.
3. Then repeat the air-thickness test (225, 450 um) at one fixed cfl with that setting. 450 um needs about 7.8 h on 4 GPUs at cfl 16, so it spans two sessions via checkpoints.
4. Identify which z wall matters: PEC on one z wall, PML on the other. This changes the domain size, so it must start from t = 0.

## Code changes
- ADI PML auxiliary checkpointing is committed by the user as `c180beb9e` ("Save checkpoint for PML aux"): `FlushFormatCheckpoint.cpp` writes `adi_psi_e0..5` and `adi_psi_h0..5`; `WarpXIO.cpp` reads them back, or warns and keeps zero if absent. There is no config setting; it is automatic when PML is on.
- Test `Examples/Tests/adi_pml_restart/analysis.py`: PASS (job 60508263). Restart energy mismatch 0.000; with psi removed 1.3e-2.
- `Bin/main3d.gnu.TPROF.MTMPI.CUDA.ex` (29 Oct) has this code. Old build kept as `Bin/...ex.bak_20261015`.
- `run_q_test/main3d.gnu.TPROF.MTMPI.CUDA.ex` is the OLD build, byte-identical to that backup, and does not save psi. It has not been swapped. Pass `EXE=` to use the new one.
- Not mine: `inputs_artemis_crosstalk` shows as modified. `run_q_test/` is untracked. Nothing of mine is uncommitted in `Source/`.

## How to run
Launch inside an existing allocation, detached on the compute node, from a login node:

    P=/pscratch/sd/r/rzhangbq
    S=/global/u2/r/rzhangbq/gt_research/AMReX/artemis/run_q_test/run_in_alloc.sh
    E=/global/u2/r/rzhangbq/gt_research/AMReX/artemis/Bin/main3d.gnu.TPROF.MTMPI.CUDA.ex
    ssh -o BatchMode=yes <node> "bash -lc 'DECK=<deck> RUN=$P/<dir>/<jobid> EXE=$E setsid nohup $S <jobid> amr.restart=<chk> max_step=<n> warpx.pml_alpha_max=<a> > $P/<dir>/launch.out 2>&1 < /dev/null & disown'"

- Add `NTASKS=1` for a single-GPU session. Decks with `air225` in the name differ by box layout: the cfl-32 one is a single box for 1 GPU, the cfl-16 one is 4 boxes.
- Batch alternative: `sbatch -t <limit> -J <name> --export=ALL,DECK=<deck>,RUNROOT=<dir> run_4gpu.sbatch [overrides]`, account m5251. The regular QOS has been waiting many hours.
- Jupyter settings for 4 GPUs: account m5251, constraint gpu, QOS jupyter, cpus-per-task 32, gpus-per-task 1, nodes 1, ntasks-per-node 4, time 360 (the maximum).
- PML inputs: `warpx.pml_R`, `warpx.pml_m`, `warpx.pml_sigma_max`, `warpx.pml_alpha_max`, `warpx.pml_kappa_max`, `warpx.pml_ncell`.

Speeds: cfl-independent per step. 37.5M cells (75 um air) 0.195 s/step on 4 GPUs; 53M cells (225 um air) 0.27 s/step on 4 GPUs, 0.74 s/step on 1 GPU. Do not put two jobs on one GPU: throughput drops sharply.

Pitfalls:
- A z PML adds 10 cells below, so `npy_k_index` is 85, not 75. x/y PML cells are already in the 680 x 460 mask. Verify the metal layer through the plotfile `mu` field; `yt` needs `ds.force_periodicity()` for the 225 um domain.
- Keep the total z cell count even (blocking factor 2). 450 um of air gives an odd count, so use 453 um or change the blocking factor.
- `flock` fails on the shared filesystem; `/tmp` is node-local, so keep scripts on `/pscratch` or in the repo.
- `pgrep -f` matches its own shell command line, so kill by explicit PID.

## Files
Under `/pscratch/sd/r/rzhangbq/`:
- `closerctrlline_cut/plot_eobs_xyz.py OUT.png RUN [RUN ...]`: Eobs in V/m, joins restarts, reports overlap mismatch.
- `cfl_study/rates.py`, `cfl_study/energy_vs_probe.py`, `cfl_study/energy_where.py`, `cfl_study/plot_cfl_shift.py`.

Runs:
- `closerctrlline_cut/60000931` + `60126164`: PEC z, 75 um air, cfl 8 (to 9.33 ns; last checkpoint `chk0225000`, no psi).
- `closerctrlline_cut_pmlall_cfl16/60211689`: all PML, 75 um air, cfl 16 (to 8.81 ns; last checkpoint `chk0100000`, no psi).
- `closerctrlline_cut_pmlall_cfl32/60570505`: all PML, 75 um air, cfl 32 (to 10 ns, with psi).
- `closerctrlline_cut_pmlall_cfl32_pmlR1e-4/60570505`: same, R = 1e-4 (to 7.98 ns).
- `closerctrlline_cut_pmlall_cfl16_air225/60595613`: 225 um air, cfl 16 (to 5.69 ns, with psi).
- `closerctrlline_cut_pmlall_cfl32_air225/60596002`: 225 um air, cfl 32 (to 4.71 ns; last checkpoint `chk0024000`, with psi).
- `air225_pmltest_cfl16_alpha0p2/60616457`: test A. `air225_pmltest_cfl32_R1e-4/60616463`: test B.

Decks are in the main checkout's `run_q_test/` (untracked): `inputs_artemis_closerctrlline_cut`, `..._pmlall_cfl8`, `..._pmlall_cfl16`, `..._pmlall_cfl32`, `..._pmlall_cfl16_air225`, `..._pmlall_cfl32_air225`.

## Full-chip estimate (8-qubit, about 10 x 10 mm, 225 um silicon + 225 um air)
- 4,020 x 4,020 x 170 = 2.75e9 cells with PML, about 73x the cut. Memory about 1.6-2.8 TB.
- To 15 ns at cfl 16: about 1,200 node-hours (about 38 h on 32 nodes). At cfl 32: about 600 node-hours, with frequencies about 2.7 % low and Q affected.
- dz = 5 um: about 720 node-hours at cfl 16, accuracy untested. Variable dz is not supported by the code.
- Parallel efficiency is a guess, about 60 %. One node gave 67 % on 4 GPUs. Multi-node scaling is unmeasured.
- Use slice diagnostics: a full plotfile would be about 150 GB.
- Blocked until the PML growth is resolved.
