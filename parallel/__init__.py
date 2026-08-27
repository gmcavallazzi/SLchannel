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

Out of scope for this prototype (next phases):
- Pencil-transposed distributed FFT for the Poisson solve (the gather-based
  version here is correct but not scalable).
- Triton kernel localization. The kernels in semilag_triton.py need: local
  extended dims as constexprs for the flat-index arrival decode, the rank
  origin (i_start, j_start) so arrival coords stay global-physical, and
  `_wrap(i, N) = i % N` (semilag_triton.py:67-70) replaced by
  `i - i_start + H` offset arithmetic with an out-of-range guard flag,
  because the constexpr NX/NY currently do double duty (arrival decode AND
  periodic wrap).
- Integration into the production SLChannelFlow driver, and multi-node.

Run the tests with:  pytest parallel/tests -q          (emulated, seconds)
                     pytest parallel/tests -q -m dist  (real 2/4-process gloo)
"""
