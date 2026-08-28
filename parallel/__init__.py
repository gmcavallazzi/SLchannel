"""z-aligned pencil decomposition prototype for slChannel.

Each rank owns a full-z pencil of the channel: the grid is split in x and y
into a px x py rank grid, and every rank stores its fields as uniform "node
arrays" of shape (nx_loc + 2H, ny_loc + 2H, NZ) with H halo layers in x and
y. The semi-Lagrangian trajectory integration and interpolation gather run
per rank with local index arithmetic (no periodic modulo: a departure foot
leaving the halo raises HaloOverflowError instead of silently aliasing a
periodic image), the z-implicit solves are column-local and reuse the
production functions unchanged, and the Poisson solve is gather-based.

Everything is validated against the monolithic solver on one GPU: the
EmulatedComm backend runs all ranks as tensors in one process, and
TorchDistComm runs real processes through torch.distributed (gloo, CPU-staged
slabs). Nothing in slchannel/ is modified; the local SL routines are
documented forks of slchannel/semilag.py pinned by a bitwise anti-drift test.

The Poisson solve has two backends: gather-based (poisson_gather.py,
simple) and pencil-transposed (poisson_pencil.py: group all-to-all
transposes keeping every FFT and tridiagonal solve local; matches the
monolithic solver exactly; all sends/recvs of a transpose are posted in a
single batch with shapes precomputed from the plan). Three transposes per
direction is structural, not sloppiness: both FFT directions need one
transpose each and the Thomas solve needs full z back, so the only way
below three is a distributed (Wang-partition) tridiagonal solve -- worth
doing only if a profile on real multi-GPU hardware shows the transposes
dominating. The Triton fast path is localized in sl_triton_local.py
(local arrival decode from the rank origin, offset arithmetic +
overflow-flag guard in place of the periodic modulo) and matches the
monolithic Triton kernels at 0 ulp.

production.py is the config-driven production driver: monolithic
operational surface (timeseries, mono-format checkpoints, snapshots, STOP
pause, statistics, blow-up guard) on the decomposed step, emulated or one
rank per process (gloo CPU-staged / NCCL device-staged), bitwise against
the monolithic solver over full runs and across restarts. Non-root ranks
build parameters-only solvers and never allocate a full-size field.
Bulk forcing is switchable: 'gathered' (bit-identical to monolithic, full
allgather per step) or 'local' (allreduced partial sums, the production
choice on real hardware).

Still out of scope: the distributed-Thomas transpose reduction (pending a
hardware profile), decomposed statistics accumulation (rank 0 still
gathers at the stats cadence), and multi-node runs.

Run the tests with:  pytest parallel/tests -q          (emulated, seconds)
                     pytest parallel/tests -q -m dist  (real 2/4-process gloo)
"""
