#!/usr/bin/env python3
"""Run with: python3 analysis.py /absolute/path/to/artemis.ex.

Check that the ADI CFS-PML convolution memory (adi_psi_*) survives a restart:
a run checkpointed at step 20 and restarted must match an uninterrupted run,
and restarting without the saved memory must not.
"""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

EXE = Path(sys.argv[1]).resolve()
NSTEPS = 40
RESTART = 20
BASE = f'''
geometry.dims = 3
geometry.prob_lo = 0 0 0
geometry.prob_hi = 0.016 0.016 0.016
amr.n_cell = 16 16 16
amr.max_level = 0
amr.max_grid_size = 64
amr.blocking_factor = 2
boundary.field_lo = pml pml pml
boundary.field_hi = pml pml pml
warpx.pml_ncell = 10
warpx.do_pml_in_domain = 0
warpx.verbose = 0
warpx.const_dt = 1.e-12
warpx.use_filter = 0
warpx.do_particle_cfl_guards = 0
algo.em_solver_medium = macroscopic
algo.time_stepping_scheme = adi
macroscopic.sigma = 0.
macroscopic.epsilon = 8.8541878128e-12
macroscopic.mu = 1.25663706212e-6
warpx.E_ext_grid_init_style = parse_E_ext_grid_function
warpx.Ex_external_grid_function(x,y,z) = "0."
warpx.Ey_external_grid_function(x,y,z) = "exp(-((x-0.004)**2 + (y-0.008)**2 + (z-0.008)**2)/(2*0.0015**2))"
warpx.Ez_external_grid_function(x,y,z) = "0."
warpx.reduced_diags_names = energy
energy.type = FieldEnergy
energy.intervals = 1
energy.path = diags/
particles.nspecies = 0
diagnostics.diags_names = chk
chk.format = checkpoint
chk.diag_type = Full
chk.intervals = {RESTART}
chk.file_min_digits = 6
'''


def run(work, extra=''):
    work.mkdir(exist_ok=True)
    (work / 'inputs').write_text(BASE + extra)
    result = subprocess.run([str(EXE), 'inputs'], cwd=work,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (work / 'run.log').write_text(result.stdout)
    assert result.returncode == 0, f'{work.name} failed; see {work / "run.log"}'
    return result.stdout


def energy(work):
    """Field energy per step, keyed by step number."""
    data = np.atleast_2d(np.loadtxt(work / 'diags' / 'energy.txt'))
    return {int(row[0]): row[2:] for row in data}


def mismatch(ref, test):
    steps = sorted(set(ref) & set(test))
    steps = [s for s in steps if s > RESTART]
    assert steps, 'no overlapping steps after the restart'
    a = np.array([ref[s] for s in steps])
    b = np.array([test[s] for s in steps])
    return float(np.max(np.abs(a - b)) / np.max(np.abs(a)))


def main():
    root = Path(tempfile.mkdtemp(prefix='adi_pml_restart_'))
    full = root / 'full'
    run(full, f'max_step = {NSTEPS}\n')
    chk = full / 'diags' / f'chk{RESTART:06d}'
    psi = sorted((chk / 'Level_0').glob('adi_psi_*_H'))
    assert len(psi) == 12, f'expected 12 adi_psi headers in {chk}, found {len(psi)}'

    restart = root / 'restart'
    run(restart, f'max_step = {NSTEPS}\namr.restart = {chk}\n')
    err = mismatch(energy(full), energy(restart))
    print(f'restart with saved PML memory: max relative energy mismatch {err:.3e}')
    assert err < 1e-12, 'restart with saved ADI PML memory does not reproduce the full run'

    # Same checkpoint without the PML memory, as written by older executables.
    old = root / 'chk_without_psi'
    shutil.copytree(chk, old)
    for path in (old / 'Level_0').glob('adi_psi_*'):
        path.unlink()
    zero = root / 'restart_zero_psi'
    log = run(zero, f'max_step = {NSTEPS}\namr.restart = {old}\n')
    err0 = mismatch(energy(full), energy(zero))
    print(f'restart without PML memory: max relative energy mismatch {err0:.3e}')
    # The warning manager wraps long messages, so match tokens that cannot be split.
    assert 'ADI PML' in log and 'adi_psi_*' in log, 'missing-memory warning not reported'
    assert err0 > 1e3 * max(err, 1e-15), 'test is insensitive: zeroing the PML memory changed nothing'

    shutil.rmtree(root)
    print('PASS')


if __name__ == '__main__':
    main()
