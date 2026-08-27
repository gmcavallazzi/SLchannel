"""Real torch.distributed backend (gloo, CPU-staged slabs), 2 and 4
processes. Opt-in: run with `-m dist` or SLC_RUN_DIST=1."""

import json
import os
import tempfile

import pytest
import torch.multiprocessing as mp

from parallel.run_dist_test import worker

pytestmark = pytest.mark.dist


@pytest.mark.parametrize("world", [2, 4])
def test_dist_backend(check, world):
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "store")
        mp.spawn(worker, args=(world, init_file, tmp), nprocs=world, join=True)
        for rank in range(world):
            with open(os.path.join(tmp, f"rank{rank}.json")) as fh:
                res = json.load(fh)
            check(
                f"w{world} r{rank} halo",
                all(res[f"halo_{c}"] == 0.0 for c in "uvw"),
                f"max halo diff {max(res[f'halo_{c}'] for c in 'uvw'):g}",
            )
            check(
                f"w{world} r{rank} advect",
                res["advect_vs_mono"] <= 1e-13,
                f"max|diff|={res['advect_vs_mono']:.3e}",
            )
            check(
                f"w{world} r{rank} reductions",
                res["allreduce_sum"] == 0.0 and res["allgather"] == 0.0,
                f"sum err {res['allreduce_sum']:g}, gather err {res['allgather']:g}",
            )
