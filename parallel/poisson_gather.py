"""Gather-based Poisson step: allgather the owned divergence blocks, run the
production FFT solver (projection.solve_poisson_fft) on the full domain, and
hand every rank its extended local slab of the pressure (halos filled from
the global periodic field, z ghosts as produced by the solver). Correct but
not scalable -- the pencil-transposed FFT is the next phase."""

from slchannel.projection import solve_poisson_fft


def solve_poisson_gathered(div_owned, comm, decomp, fft_data, dt_eff):
    """div_owned: dict rank -> (nxl, nyl, nz) owned divergence. Returns
    dict rank -> extended local pressure array (node layout, z len nz+2)."""
    full = comm.allgather_nodes(div_owned)
    # every entry of `full` is the same tensor per local rank; solve once
    any_rank = comm.local_ranks[0]
    p_mono = solve_poisson_fft(full[any_rank] / dt_eff, fft_data)
    nx, ny = decomp.nx, decomp.ny
    p_nodes = p_mono[1 : nx + 1, 1 : ny + 1, :].contiguous()
    scattered = decomp.scatter(p_nodes, "p", fill_halos=True)
    return {r: scattered[r] for r in comm.local_ranks}
