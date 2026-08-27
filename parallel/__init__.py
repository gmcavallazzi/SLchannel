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
monolithic solver exactly, three transposes per direction -- a production
version would restructure to two or adopt cuDecomp). The Triton fast path
is localized in sl_triton_local.py (local arrival decode from the rank
origin, offset arithmetic + overflow-flag guard in place of the periodic
modulo) and matches the monolithic Triton kernels at 0 ulp.

Still out of scope: integration into the production SLChannelFlow driver,
two-transpose Poisson restructuring, and multi-node runs.

Run the tests with:  pytest parallel/tests -q          (emulated, seconds)
                     pytest parallel/tests -q -m dist  (real 2/4-process gloo)
"""
