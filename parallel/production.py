"""Production driver for the decomposed solver: a config-driven run loop
with the monolithic solver's operational surface (timeseries.npz, fields.npz
checkpoints, t-snapshots, STOP-file pause, statistics state, blow-up guard)
on top of DecomposedBDF2.

Launch, single process with all ranks emulated (any device):

    python -m parallel.production configs/case.yaml --px 2 --py 2

or one process per rank through torch.distributed (gloo by default; NCCL on
a single shared GPU needs CUDA MPS and SLC_DIST_BACKEND=nccl):

    torchrun --nproc_per_node=4 -m parallel.production configs/case.yaml \
        --px 2 --py 2 --backend dist

Semantics and deliberate differences from SLChannelFlow.run_simulation:

- dt is CONSTANT: `time.dt` is used as-is and `dt_update_interval` is
  ignored with a warning (BDF2 needs constant dt; pin the production value
  in the config).
- Initialization runs on rank 0 exactly like the monolithic solver
  (including the divergence-free projection and fields_init.npz), then the
  field is scattered; under `dist` the other ranks build parameters-only
  solvers and never allocate a full-size field. Rank 0 still materializes
  full fields transiently during seeding and gathers (statistics,
  checkpoints) — moving those off rank 0 is future work.
- Diagnostics (max div, u_tau, CFL) and statistics run on rank 0 on the
  gathered field with the production operators, so the numbers are
  bit-identical to a monolithic run of the same trajectory.
- Checkpoints are the monolithic fields.npz format: a decomposed run can be
  restarted monolithically and vice versa (BDF2 re-bootstraps either way).
"""

import argparse
import os
import sys
import time as walltime

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from parallel.comm import EmulatedComm, TorchDistComm  # noqa: E402
from parallel.decomp import Decomp, mono_node_view  # noqa: E402
from parallel.sl_local import required_halo  # noqa: E402
from parallel.step import DecomposedBDF2  # noqa: E402
from slchannel import operators  # noqa: E402
from slchannel.solver import SLChannelFlow  # noqa: E402
from slchannel.utils import (  # noqa: E402
    compute_divergence,
    compute_u_tau,
    save_flow_fields,
)

# displacement growth allowance between the initial field (which sizes the
# halo) and the developed flow; HaloOverflowError still guards hard.
HALO_SAFETY = 1.3


def _is_dist():
    import torch.distributed as dist

    return dist.is_available() and dist.is_initialized()


def _bcast_scalar(x):
    """Root's float scalar to every rank (identity when not distributed)."""
    if not _is_dist():
        return float(x)
    import torch.distributed as dist

    t = torch.tensor([float(x) if dist.get_rank() == 0 else 0.0], dtype=torch.float64)
    dist.broadcast(t, src=0)
    return float(t.item())


class ProductionRun:
    def __init__(self, config_file, px, py, backend="emulated", poisson="pencil", triton=None):
        self.backend = backend
        self.is_root = True
        rank = 0
        if backend == "dist":
            import torch.distributed as dist

            assert dist.is_initialized(), (
                "backend 'dist' needs an initialized process group (launch via torchrun)"
            )
            rank = dist.get_rank()
            self.is_root = rank == 0
            assert dist.get_world_size() == px * py, (
                f"world size {dist.get_world_size()} != px*py = {px * py}"
            )
            if torch.cuda.is_available():
                torch.cuda.set_device(rank % torch.cuda.device_count())

        # rank 0 initializes exactly like the monolithic solver (projection,
        # fields_init.npz, grid.csv); the others are parameters-only
        self.mono = SLChannelFlow(config_file, allocate_fields=self.is_root)
        mono = self.mono
        self.dt = mono.dt
        if mono.dt_update_interval > 0 and self.is_root:
            print(
                f"[production] dt_update_interval={mono.dt_update_interval} is ignored: "
                f"the decomposed driver holds dt = {self.dt:g} constant",
                flush=True,
            )

        # halo width from the initial field's maximum displacement
        umax = 0.0
        if self.is_root:
            umax = float(
                torch.stack([mono.u.abs().max(), mono.v.abs().max(), mono.w.abs().max()]).max()
            )
        umax = _bcast_scalar(umax)
        disp = HALO_SAFETY * self.dt * umax / min(mono.dx, mono.dy)
        H = required_halo(mono.sl.order, disp_cells=disp)

        self.d = Decomp(px, py, mono.nx, mono.ny, mono.nz, H=H)
        if backend == "dist":
            self.comm = TorchDistComm(self.d, device=mono.device)
        else:
            self.comm = EmulatedComm(self.d, device=mono.device)

        use_triton = mono.device.type == "cuda" if triton is None else triton
        seed = None
        if self.is_root:
            seed = {
                c: mono_node_view(getattr(mono, c), c, mono.nx, mono.ny).contiguous() for c in "uvw"
            }
            # the driver owns the fields from here; free the monolithic copies
            mono.u = mono.v = mono.w = mono.p = None
        self.dec = DecomposedBDF2(mono, self.d, self.comm, poisson=poisson, use_triton=use_triton)
        self.dec.set_state_nodes(seed if seed is not None else {})
        del seed

        self.time = _bcast_scalar(mono.initial_time)
        self.step = int(_bcast_scalar(mono.initial_step))
        if self.is_root:
            print(
                f"[production] {px}x{py} ranks ({backend}), H={H} "
                f"(disp {disp:.2f} cells/dt incl. x{HALO_SAFETY} margin, "
                f"order {mono.sl.order}), poisson={poisson}, triton={use_triton}",
                flush=True,
            )

        # rank-0 run bookkeeping
        self._gather_cache = (None, None)  # (step, (u, v, w))
        self._blowup_strikes = 0
        delta = mono.Lz if mono.top_wall_bc_type == "neumann" else mono.Lz / 2.0
        self._u_tau_nominal = mono.Re_tau * mono.nu / delta
        chunk = mono.n_save // mono.n_out + 1
        self._ts = {
            k: np.zeros(chunk, dtype=np.int32 if k == "step" else np.float64)
            for k in ("step", "time", "u_bulk", "u_tau", "forcing", "cfl")
        }
        self._ts_index = 0

    # ---- rank-0 helpers ----------------------------------------------------

    def _gather_uvw(self):
        """Collective; returns the gathered (u, v, w) on every rank, cached
        per step so coinciding cadences gather once."""
        if self._gather_cache[0] == self.step:
            return self._gather_cache[1]
        uvw = tuple(self.dec.gather_mono(c) for c in "uvw")
        self._gather_cache = (self.step, uvw)
        return uvw

    def _flush_timeseries(self):
        if self._ts_index == 0:
            return
        npz_file = os.path.join(self.mono.results_folder, "timeseries.npz")
        chunks = {k: v[: self._ts_index] for k, v in self._ts.items()}
        if os.path.exists(npz_file):
            existing = np.load(npz_file)
            chunks = {k: np.concatenate([existing[k], chunks[k]]) for k in chunks}
        np.savez_compressed(npz_file, **chunks)
        self._ts_index = 0

    def _save_fields(self, name, u_tau, forcing):
        u, v, w = self._gather_uvw()
        p = self.dec.gather_mono("p")
        if not self.is_root:
            return
        if p is None:
            p = torch.zeros(
                self.mono.nx + 2,
                self.mono.ny + 2,
                self.mono.nz + 2,
                dtype=u.dtype,
                device=u.device,
            )
        save_flow_fields(
            u,
            v,
            w,
            p,
            self.mono.z_c,
            self.mono.z_f,
            self.mono.Lx,
            self.mono.Ly,
            self.step,
            self.time,
            u_tau,
            forcing,
            self.mono.results_folder,
            name,
        )

    def _blowup_check(self, u_tau):
        if u_tau > self.mono.blowup_u_tau_factor * self._u_tau_nominal:
            self._blowup_strikes += 1
        else:
            self._blowup_strikes = 0
        return self._blowup_strikes >= 3

    # ---- the loop ----------------------------------------------------------

    def run(self):
        mono = self.mono
        dt = self.dt
        stop_path = os.path.join(mono.results_folder, "STOP")
        next_snap = None
        if mono.t_snapshot > 0:
            next_snap = self.time + mono.t_snapshot
        u_tau_scalar = forcing_scalar = 0.0
        stopped = blown = False
        t0 = walltime.time()

        if self.is_root:
            print(
                f"{'Step':>6} {'Time':>10} {'max(div)':>12} {'u_bulk':>10} "
                f"{'u_tau':>10} {'forcing':>12} {'CFL':>8}",
                flush=True,
            )

        while self.step < mono.n_steps and self.time < mono.t_max:
            self.step += 1
            forcing_t = self.dec.step(dt)
            self.time += dt
            forcing_scalar = float(forcing_t)
            u_bulk_scalar = mono.U_bulk - forcing_scalar * dt

            stop_flag = 0.0
            if self.step % mono.n_out == 0:
                u, v, w = self._gather_uvw()
                if self.is_root:
                    div = compute_divergence(
                        u, v, w, mono.nx, mono.ny, mono.nz, mono.dx, mono.dy, mono.dz_f
                    )
                    max_div = float(div.abs().max())
                    u_tau_scalar = float(
                        compute_u_tau(u, mono.z_c, mono.nu, top_wall_bc_type=mono.top_wall_bc_type)
                    )
                    cfl = (
                        float(
                            operators.compute_cfl_fused(
                                u,
                                v,
                                w,
                                mono.nx,
                                mono.ny,
                                mono.nz,
                                mono.dx,
                                mono.dy,
                                mono.dz_f,
                                mono.dz_c,
                            )
                        )
                        * dt
                    )
                    idx = self._ts_index
                    for k, val in (
                        ("step", self.step),
                        ("time", self.time),
                        ("u_bulk", u_bulk_scalar),
                        ("u_tau", u_tau_scalar),
                        ("forcing", forcing_scalar),
                        ("cfl", cfl),
                    ):
                        self._ts[k][idx] = val
                    self._ts_index += 1
                    print(
                        f"{self.step:6d} {self.time:10.6f} {max_div:12.3e} "
                        f"{u_bulk_scalar:10.6f} {u_tau_scalar:10.6f} "
                        f"{forcing_scalar:12.3e} {cfl:8.3f}  "
                        f"[{(walltime.time() - t0) / max(1, self.step - int(mono.initial_step)):.3f} s/step]",
                        flush=True,
                    )
                    if mono.stop_on_blowup and self._blowup_check(u_tau_scalar):
                        print(
                            f"BLOW-UP: u_tau = {u_tau_scalar:.4f} on three consecutive "
                            f"diagnostics at step {self.step} -- stopping",
                            flush=True,
                        )
                        stop_flag = 2.0
                    elif os.path.exists(stop_path):
                        print(
                            f"STOP file found -- checkpointing and stopping at "
                            f"step {self.step}, time = {self.time:.6f}",
                            flush=True,
                        )
                        stop_flag = 1.0
                stop_flag = float(
                    self.comm.allreduce({r: stop_flag for r in self.comm.local_ranks}, op="max")[
                        self.comm.local_ranks[0]
                    ]
                )

            if mono.n_stats > 0 and self.time >= mono.t_stats and self.step % mono.n_stats == 0:
                u, v, w = self._gather_uvw()
                if self.is_root:
                    u_tau_stats = compute_u_tau(
                        u, mono.z_c, mono.nu, top_wall_bc_type=mono.top_wall_bc_type
                    )
                    if mono.turbulence_stats.n_samples == 0:
                        print(f"  [Stats] starting collection at t = {self.time:.3f}", flush=True)
                    mono.turbulence_stats.accumulate_statistics(u, v, w, u_tau_stats)

            checkpoint_due = self.step % mono.n_save == 0 or stop_flag > 0.0
            if checkpoint_due:
                self._save_fields("fields.npz", u_tau_scalar, forcing_scalar)
                if self.is_root:
                    self._flush_timeseries()
                    if mono.turbulence_stats is not None and mono.turbulence_stats.n_samples > 0:
                        mono.turbulence_stats.save_state(mono.stats_state_path)

            if next_snap is not None and self.time >= next_snap - 1e-12:
                while next_snap <= self.time + 1e-12:
                    next_snap += mono.t_snapshot
                name = f"fields_t{self.time:09.3f}.npz"
                self._save_fields(name, u_tau_scalar, forcing_scalar)
                if self.is_root:
                    print(f"  [Snapshot] Saved {name}", flush=True)

            if stop_flag > 0.0:
                stopped = stop_flag == 1.0
                blown = stop_flag == 2.0
                break

        completed = not stopped and not blown
        if completed:
            self._save_fields("fields_final.npz", u_tau_scalar, forcing_scalar)
        if self.is_root:
            self._flush_timeseries()
            if mono.turbulence_stats is not None and mono.turbulence_stats.n_samples > 0:
                mono.turbulence_stats.save_statistics(mono.stats_output_path)
                mono.turbulence_stats.save_state(mono.stats_state_path)
            outcome = "paused (STOP file)" if stopped else ("blown up" if blown else "complete")
            print(
                f"[production] {outcome}: {self.step} steps, t = {self.time:.6f}, "
                f"{walltime.time() - t0:.1f} s wall",
                flush=True,
            )
        return {"step": self.step, "time": self.time, "stopped": stopped, "blown": blown}


def _dist_test_worker(rank, world, init_file, cfg_path, px, py):
    """Spawn target for tests/test_dist_production.py (gloo, file store)."""
    import torch.distributed as dist

    torch.set_default_dtype(torch.float64)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        run = ProductionRun(cfg_path, px, py, backend="dist", poisson="pencil", triton=False)
        run.run()
    finally:
        dist.destroy_process_group()


def main(argv=None):
    ap = argparse.ArgumentParser(description="decomposed production run")
    ap.add_argument("config")
    ap.add_argument("--px", type=int, required=True)
    ap.add_argument("--py", type=int, required=True)
    ap.add_argument("--backend", choices=["emulated", "dist"], default="emulated")
    ap.add_argument("--poisson", choices=["pencil", "gathered"], default="pencil")
    ap.add_argument("--no-triton", action="store_true")
    args = ap.parse_args(argv)

    if args.backend == "dist":
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(os.environ.get("SLC_DIST_BACKEND", "gloo"))

    run = ProductionRun(
        args.config,
        args.px,
        args.py,
        backend=args.backend,
        poisson=args.poisson,
        triton=False if args.no_triton else None,
    )
    run.run()


if __name__ == "__main__":
    main()
