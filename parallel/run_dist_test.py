"""torch.distributed worker for the decomposition tests.

Spawned by tests/test_dist_backend.py (gloo, file:// store, CPU tensors), or
runnable standalone for manual experiments:

    torchrun --nproc_per_node=2 parallel/run_dist_test.py

NCCL on a single shared GPU requires CUDA MPS and is opt-in via
SLC_DIST_BACKEND=nccl; never start MPS while a production run owns the card.
"""

import json
import os
import sys

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "parallel", "tests"))


def worker(rank, world, init_file, out_dir):
    import torch.distributed as dist

    torch.set_default_dtype(torch.float64)  # spawned procs do not inherit fixtures
    backend = os.environ.get("SLC_DIST_BACKEND", "gloo")
    dist.init_process_group(backend, init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        results = _run_checks(rank, world)
        with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as fh:
            json.dump(results, fh)
    finally:
        dist.destroy_process_group()


def _run_checks(rank, world):
    from helpers_par import build_ref, default_fields, mono_advect_nodes

    from parallel.comm import EmulatedComm, TorchDistComm
    from parallel.decomp import Decomp
    from parallel.sl_local import LocalSL, required_halo

    px, py = (2, 2) if world == 4 else (world, 1)
    order = 4
    H = required_halo(order, disp_cells=1.0, foot_depth_factor=1.0)
    ref, grid = build_ref(nx=24, ny=24, nz=12, order=order)
    d = Decomp(px, py, 24, 24, 12, H=H)
    comm = TorchDistComm(d)
    nodes = default_fields(grid, seed=0)
    mids = default_fields(grid, seed=1)

    results = {}

    # 1. halo exchange bitwise vs the emulated backend
    emu = EmulatedComm(d)
    for comp in "uvw":
        mine = {rank: d.scatter(nodes[comp], comp, fill_halos=False)[rank]}
        comm.halo_exchange(mine)
        ref_locs = d.scatter(nodes[comp], comp, fill_halos=False)
        emu.halo_exchange(ref_locs)
        results[f"halo_{comp}"] = float((mine[rank] - ref_locs[rank]).abs().max())

    # 2. distributed local advect vs the monolithic advect
    dt_t = torch.as_tensor(0.4 * grid["dx"], dtype=torch.float64)
    mono = mono_advect_nodes(ref, grid, nodes, mids, dt_t)
    sl = LocalSL(ref, d, rank)
    ext = {c: d.scatter(nodes[c], c, fill_halos=True)[rank] for c in "uvw"}
    mid = {c: d.scatter(mids[c], c, fill_halos=True)[rank] for c in "uvw"}
    outs = sl.advect(ext, mid, dt_t)
    nxl, nyl, nz = d.nxl, d.nyl, d.nz
    i0, j0 = d.origin(rank)
    err = 0.0
    # compare at this rank's arrival nodes (see sl_local docstring: the u/x
    # and v/y face sets are shifted by one relative to storage ownership)
    gx_u = (torch.arange(i0 + 1, i0 + nxl + 1)) % d.nx
    gy_u = torch.arange(j0, j0 + nyl)
    err = max(err, float((outs["u"] - mono["u"][gx_u][:, gy_u, 1 : nz + 1]).abs().max()))
    gx_v = torch.arange(i0, i0 + nxl)
    gy_v = (torch.arange(j0 + 1, j0 + nyl + 1)) % d.ny
    err = max(err, float((outs["v"] - mono["v"][gx_v][:, gy_v, 1 : nz + 1]).abs().max()))
    err = max(err, float((outs["w"] - mono["w"][gx_v][:, gy_u, 1:nz]).abs().max()))
    results["advect_vs_mono"] = err

    # 3. allreduce and allgather round trips
    red = comm.allreduce({rank: torch.tensor(float(rank + 1))}, op="sum")[rank]
    results["allreduce_sum"] = float(red) - world * (world + 1) / 2.0
    full = comm.allgather_nodes({rank: ext["w"]})[rank]
    results["allgather"] = float((full - nodes["w"]).abs().max())
    return results


def main():
    import torch.distributed as dist

    dist.init_process_group(os.environ.get("SLC_DIST_BACKEND", "gloo"))
    torch.set_default_dtype(torch.float64)
    results = _run_checks(dist.get_rank(), dist.get_world_size())
    print(f"[rank {dist.get_rank()}] {results}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
