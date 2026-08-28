"""Pencil-transposed distributed Poisson solve.

Replaces the gather-based solve with group all-to-all transposes on the
(px, py) rank grid, keeping every FFT and the tridiagonal solve local:

    z-pencil (nxl, nyl, nz)                 owned divergence
      --[A2A in the py column group, splitting z]-->
    y-pencil (nxl, ny, nz/py)               rfft along y -> (nxl, nky, nz/py)
      --[A2A in the px row group, splitting ky (uneven: nky = ny//2+1)]-->
    x-pencil (nx, nky_c, nz/py)             complex fft along x
      --[A2A in the py column group, splitting x]-->
    spectral z-pencil (nx/py, nky_c, nz)    production Thomas solve on sliced
                                            tri_a/tri_b/tri_c coefficients
      --[inverse chain]-->                  owned pressure (nxl, nyl, nz)

Numerics: torch.fft.rfft2(dim=(0, 1)) is replaced by rfft(y) followed by
fft(x) -- mathematically identical, so the result matches the monolithic
solver to rounding (~1e-12 relative), not bitwise. Three transposes per
direction is STRUCTURAL from a z-pencil start: each FFT direction needs
one transpose and the Thomas solve needs full z back; the only way below
three is a distributed (Wang-partition) tridiagonal solve, worth doing
only if a hardware profile shows the transposes dominating. All
sends/recvs of a transpose are posted in one batch with shapes
precomputed from the plan. Constraints: nz % py == 0 and nx % py == 0
(asserted).

Complex tensors are moved through the communicator as real (..., 2) views so
the gloo backend never sees complex dtypes.
"""

import torch

from slchannel.projection import solve_tridiagonal


def _chunks(n, parts):
    """Contiguous split of n into `parts` chunks (first chunks get the
    remainder); returns list of (offset, count)."""
    base, rem = divmod(n, parts)
    out, off = [], 0
    for i in range(parts):
        c = base + (1 if i < rem else 0)
        out.append((off, c))
        off += c
    return out


class PencilPlan:
    def __init__(self, decomp):
        d = decomp
        assert d.nz % d.py == 0, "pencil Poisson needs nz % py == 0"
        assert d.nx % d.py == 0, "pencil Poisson needs nx % py == 0"
        self.d = d
        self.nky = d.ny // 2 + 1
        self.nzb = d.nz // d.py  # z chunk per column-group member
        self.nxs = d.nx // d.py  # x chunk per column-group member (spectral)
        self.ky_chunks = _chunks(self.nky, d.px)  # uneven ky split per row rank

    def col_group(self, rank):
        cx, _ = self.d.coords(rank)
        return [self.d.rank_of(cx, j) for j in range(self.d.py)]

    def row_group(self, rank):
        _, cy = self.d.coords(rank)
        return [self.d.rank_of(i, cy) for i in range(self.d.px)]

    def kyc(self, rank):
        """This rank's ky-chunk length (set by its row position)."""
        cx, _ = self.d.coords(rank)
        return self.ky_chunks[cx][1]


def _exchange(comm, sendmaps, recv_shapes=None):
    """sendmaps: dict src_rank -> dict dst_rank -> tensor. Returns dict
    dst_rank -> dict src_rank -> tensor (local ranks only). Complex tensors
    are staged as real views for the distributed backend.

    recv_shapes: dict local_rank -> dict src_rank -> tuple, the tensor shape
    each peer will send (blocks can be shape-asymmetric through the uneven
    ky chunks). When given, every send/recv is posted in ONE
    batch_isend_irecv — no per-peer serialization. Without it, a shape
    header handshake runs per peer (kept as the fallback)."""
    if hasattr(comm, "dist"):  # TorchDistComm
        dist, me = comm.dist, comm.rank
        my = sendmaps[me]
        recv = {}
        peers = sorted(my)
        if recv_shapes is not None:
            shapes = recv_shapes[me]
            ops, staged, backs, complexes = [], {}, {}, {}
            for peer in peers:
                t = my[peer]
                complexes[peer] = t.is_complex()
                send = (
                    (torch.view_as_real(t) if complexes[peer] else t)
                    .contiguous()
                    .to(comm.stage_device)
                )
                if peer == me:
                    backs[peer] = send.clone()
                    continue
                staged[peer] = send
                shp = tuple(shapes[peer]) + ((2,) if complexes[peer] else ())
                backs[peer] = torch.empty(shp, dtype=send.dtype, device=comm.stage_device)
                ops.append(dist.P2POp(dist.isend, send, peer, tag=comm._tag(78)))
                ops.append(dist.P2POp(dist.irecv, backs[peer], peer, tag=comm._tag(78)))
            if ops:
                for req in dist.batch_isend_irecv(ops):
                    req.wait()
            for peer in peers:
                back = backs[peer].to(my[peer].device)
                recv[peer] = torch.view_as_complex(back) if complexes[peer] else back
            return {me: recv}
        for peer in peers:
            t = my[peer]
            is_c = t.is_complex()
            send = (torch.view_as_real(t) if is_c else t).contiguous().to(comm.stage_device)
            if peer == me:
                back = send.clone()
            else:
                # shape header handshake, then the payload
                hdr_out = torch.tensor(
                    list(send.shape), dtype=torch.int64, device=comm.stage_device
                )
                hdr_in = torch.empty_like(hdr_out)
                ops = [
                    dist.P2POp(dist.isend, hdr_out, peer, tag=comm._tag(76)),
                    dist.P2POp(dist.irecv, hdr_in, peer, tag=comm._tag(76)),
                ]
                for req in dist.batch_isend_irecv(ops):
                    req.wait()
                back = torch.empty(
                    tuple(hdr_in.tolist()), dtype=send.dtype, device=comm.stage_device
                )
                ops = [
                    dist.P2POp(dist.isend, send, peer, tag=comm._tag(77)),
                    dist.P2POp(dist.irecv, back, peer, tag=comm._tag(77)),
                ]
                for req in dist.batch_isend_irecv(ops):
                    req.wait()
            back = back.to(t.device)
            recv[peer] = torch.view_as_complex(back) if is_c else back
        return {me: recv}
    # EmulatedComm: pure shuffle
    out = {r: {} for r in sendmaps}
    for src, m in sendmaps.items():
        for dst, t in m.items():
            out[dst][src] = t.clone()
    return out


def _fwd(plan, comm, div_local):
    """Owned divergence blocks -> spectral z-pencil blocks per local rank.
    Returns dict rank -> (nxs, nky_c, nz) complex."""
    d = plan.d
    # 1. z -> y within the column group (peer j takes z chunk j)
    send = {}
    for r, blk in div_local.items():
        cg = plan.col_group(r)
        send[r] = {peer: blk[:, :, j * plan.nzb : (j + 1) * plan.nzb] for j, peer in enumerate(cg)}
    shapes = {r: {peer: (d.nxl, d.nyl, plan.nzb) for peer in plan.col_group(r)} for r in div_local}
    recv = _exchange(comm, send, shapes)
    ypen = {}
    for r in recv:
        cg = plan.col_group(r)
        ypen[r] = torch.cat([recv[r][peer] for peer in cg], dim=1)  # (nxl, ny, nzb)

    # 2. real FFT along y, then 3. y -> x within the row group (ky chunks)
    send = {}
    for r, blk in ypen.items():
        fy = torch.fft.rfft(blk, dim=1)
        rg = plan.row_group(r)
        send[r] = {
            peer: fy[:, off : off + cnt, :].contiguous()
            for (off, cnt), peer in zip(plan.ky_chunks, rg)
        }
    shapes = {r: {peer: (d.nxl, plan.kyc(r), plan.nzb) for peer in plan.row_group(r)} for r in ypen}
    recv = _exchange(comm, send, shapes)
    xpen = {}
    for r in recv:
        rg = plan.row_group(r)
        xpen[r] = torch.cat([recv[r][peer] for peer in rg], dim=0)  # (nx, nky_c, nzb)

    # 4. complex FFT along x, then 5. spectral z regather (x chunks)
    send = {}
    for r, blk in xpen.items():
        fx = torch.fft.fft(blk, dim=0)
        cg = plan.col_group(r)
        send[r] = {
            peer: fx[j * plan.nxs : (j + 1) * plan.nxs, :, :].contiguous()
            for j, peer in enumerate(cg)
        }
    shapes = {
        r: {peer: (plan.nxs, plan.kyc(r), plan.nzb) for peer in plan.col_group(r)} for r in xpen
    }
    recv = _exchange(comm, send, shapes)
    spec = {}
    for r in recv:
        cg = plan.col_group(r)
        spec[r] = torch.cat([recv[r][peer] for peer in cg], dim=2)  # (nxs, nky_c, nz)
    return spec


def _bwd(plan, comm, spec):
    """Inverse of _fwd: spectral z-pencil -> owned real blocks."""
    d = plan.d
    # 5'. z scatter back to x-pencils
    send = {}
    for r, blk in spec.items():
        cg = plan.col_group(r)
        send[r] = {
            peer: blk[:, :, j * plan.nzb : (j + 1) * plan.nzb].contiguous()
            for j, peer in enumerate(cg)
        }
    shapes = {
        r: {peer: (plan.nxs, plan.kyc(r), plan.nzb) for peer in plan.col_group(r)} for r in spec
    }
    recv = _exchange(comm, send, shapes)
    xpen = {}
    for r in recv:
        cg = plan.col_group(r)
        xpen[r] = torch.cat([recv[r][peer] for peer in cg], dim=0)  # (nx, nky_c, nzb)

    # 4'. inverse complex FFT along x, then 3'. x -> y (return the ky chunks)
    send = {}
    for r, blk in xpen.items():
        fx = torch.fft.ifft(blk, dim=0)
        rg = plan.row_group(r)
        send[r] = {
            peer: fx[d.coords(peer)[0] * d.nxl : (d.coords(peer)[0] + 1) * d.nxl, :, :].contiguous()
            for peer in rg
        }
    shapes = {
        r: {peer: (d.nxl, plan.kyc(peer), plan.nzb) for peer in plan.row_group(r)} for r in xpen
    }
    recv = _exchange(comm, send, shapes)
    ypen = {}
    for r in recv:
        rg = plan.row_group(r)
        ypen[r] = torch.cat(
            [recv[r][peer] for peer in rg], dim=1
        )  # (nxl, nky, nzb) -- ky chunks back in row order == contiguous ky
    # 2'. inverse real FFT along y, then 1'. y -> z (return the z chunks)
    send = {}
    for r, blk in ypen.items():
        ry = torch.fft.irfft(blk, n=d.ny, dim=1)
        cg = plan.col_group(r)
        send[r] = {
            peer: ry[:, d.coords(peer)[1] * d.nyl : (d.coords(peer)[1] + 1) * d.nyl, :].contiguous()
            for peer in cg
        }
    shapes = {r: {peer: (d.nxl, d.nyl, plan.nzb) for peer in plan.col_group(r)} for r in ypen}
    recv = _exchange(comm, send, shapes)
    out = {}
    for r in recv:
        cg = plan.col_group(r)
        out[r] = torch.cat([recv[r][peer] for peer in cg], dim=2)  # (nxl, nyl, nz)
    return out


def solve_poisson_pencil(div_owned, comm, decomp, fft_data, dt_eff, H=None):
    """Same contract as poisson_gather.solve_poisson_gathered: dict rank ->
    (nxl, nyl, nz) divergence in, dict rank -> extended local pressure slab
    out (halos exchanged, Neumann z ghosts)."""
    d = decomp
    plan = PencilPlan(d)
    rhs = {r: blk / dt_eff for r, blk in div_owned.items()}
    spec = _fwd(plan, comm, rhs)

    # local tridiagonal solve on production coefficients, sliced to this
    # rank's (kx chunk, ky chunk) block (projection.py:137-164 equivalent)
    for r, blk in spec.items():
        cx, cy = d.coords(r)
        kx0 = cy * plan.nxs  # x chunk index within the column group
        ky0, kyc = plan.ky_chunks[cx]
        ta = fft_data["tri_a"][kx0 : kx0 + plan.nxs, ky0 : ky0 + kyc, :]
        tb = fft_data["tri_b"][kx0 : kx0 + plan.nxs, ky0 : ky0 + kyc, :]
        tc = fft_data["tri_c"][kx0 : kx0 + plan.nxs, ky0 : ky0 + kyc, :]
        nb = plan.nxs * kyc
        sol = solve_tridiagonal(
            ta.reshape(nb, d.nz),
            tb.reshape(nb, d.nz),
            tc.reshape(nb, d.nz),
            blk.reshape(nb, d.nz),
        )
        spec[r] = sol.reshape(plan.nxs, kyc, d.nz)

    p_owned = _bwd(plan, comm, spec)

    # assemble extended slabs: interior + halo exchange + Neumann z ghosts
    # (matching solve_poisson_fft's ghost fills, projection.py:182-195)
    ext = {}
    for r, blk in p_owned.items():
        e = d.alloc("p", dtype=blk.dtype, device=blk.device)
        e[d.H : d.H + d.nxl, d.H : d.H + d.nyl, 1 : d.nz + 1] = blk
        e[:, :, 0] = e[:, :, 1]
        e[:, :, d.nz + 1] = e[:, :, d.nz]
        ext[r] = e
    comm.halo_exchange(ext)
    # halo columns carry interior data but stale z ghosts; refresh locally
    for e in ext.values():
        e[:, :, 0] = e[:, :, 1]
        e[:, :, d.nz + 1] = e[:, :, d.nz]
    return ext
