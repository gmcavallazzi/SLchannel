"""Halo slab bookkeeping shared by both communicator backends.

Exchange runs as two passes: x first over the full y extent, then y over the
full x extent (which by then includes the freshly filled x halos), so the
corner halos arrive without explicit diagonal messages. All slices are in the
extended local layout of decomp.Decomp.
"""


def _sl(dim, a, b):
    """Slice tuple selecting [a:b) along dim (0 or 1), everything else full."""
    s = [slice(None), slice(None), slice(None)]
    s[dim] = slice(a, b)
    return tuple(s)


def send_recv_slices(n_loc, H, dim):
    """(send_to_minus, send_to_plus, recv_from_minus, recv_from_plus) slice
    tuples for one exchange direction.

    The minus neighbor's halo shows my leading owned slab [H, 2H); the plus
    neighbor's halo shows my trailing owned slab [n_loc, n_loc+H). My own
    halos are [0, H) (from minus) and [n_loc+H, n_loc+2H) (from plus)."""
    send_m = _sl(dim, H, 2 * H)
    send_p = _sl(dim, n_loc, n_loc + H)
    recv_m = _sl(dim, 0, H)
    recv_p = _sl(dim, n_loc + H, n_loc + 2 * H)
    return send_m, send_p, recv_m, recv_p


def edge_pull_slices(n_loc, H, dim):
    """(dst, src) slices for the ownership pull of the shared staggered face:
    my owned leading plane [H] is authoritative in my MINUS neighbor's slab
    [n_loc+H] after an SL advect (u in x, v in y) — I pull it from there."""
    return _sl(dim, H, H + 1), _sl(dim, n_loc + H, n_loc + H + 1)
