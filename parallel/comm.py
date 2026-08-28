"""Communicator backends: emulated single-process ranks and torch.distributed.

Both expose the same call surface over `fields: dict[int, Tensor]` mapping
rank -> extended local array. EmulatedComm holds every rank's tensor in one
process (exchange = slice copies, deterministic, fast-suite friendly);
TorchDistComm holds exactly one rank per process and moves the identical
slabs through torch.distributed: gloo stages through contiguous CPU
tensors (correctness on any device), NCCL stages on the CUDA device
itself. NCCL on a single shared GPU needs CUDA MPS and is opt-in via
SLC_DIST_BACKEND=nccl."""

import torch

from .halo import edge_pull_slices, send_recv_slices


class EmulatedComm:
    """All ranks in one process; `fields` carries every rank's tensor."""

    def __init__(self, decomp, device="cpu"):
        self.decomp = decomp
        self.nranks = decomp.nranks
        self.local_ranks = list(range(decomp.nranks))
        self.device = torch.device(device)

    def _exchange_dim(self, fields, dim, n_loc):
        d, H = self.decomp, self.decomp.H
        send_m, send_p, recv_m, recv_p = send_recv_slices(n_loc, H, dim)
        # snapshot the owned slabs first so a self-neighbor (px or py == 1)
        # exchange reads pre-exchange data, like a real communicator would
        slabs = {r: (fields[r][send_m].clone(), fields[r][send_p].clone()) for r in fields}
        for r in fields:
            nb = d.neighbors(r)
            minus = nb["xm"] if dim == 0 else nb["ym"]
            plus = nb["xp"] if dim == 0 else nb["yp"]
            fields[r][recv_m] = slabs[minus][1]  # minus neighbor's trailing slab
            fields[r][recv_p] = slabs[plus][0]  # plus neighbor's leading slab

    def halo_exchange(self, fields):
        self._exchange_dim(fields, 0, self.decomp.nxl)
        self._exchange_dim(fields, 1, self.decomp.nyl)

    def pull_minus_edge(self, fields, dim):
        d = self.decomp
        n_loc = d.nxl if dim == 0 else d.nyl
        dst, src = edge_pull_slices(n_loc, d.H, dim)
        slabs = {r: fields[r][src].clone() for r in fields}
        for r in fields:
            nb = d.neighbors(r)
            minus = nb["xm"] if dim == 0 else nb["ym"]
            fields[r][dst] = slabs[minus]

    def scatter_nodes(self, nodes, comp):
        """Distribute a full monolithic node array (present on the root; may
        be None elsewhere) into per-rank extended arrays with halos filled."""
        full = self.decomp.scatter(nodes.to(self.device), comp, fill_halos=True)
        return {r: full[r] for r in self.local_ranks}

    def allgather_nodes(self, fields):
        """Assemble the global node array from owned regions; every local
        rank sees the same tensor."""
        d, H = self.decomp, self.decomp.H
        r0 = fields[self.local_ranks[0]]
        full = torch.zeros(d.nx, d.ny, r0.shape[2], dtype=r0.dtype, device=r0.device)
        for rank, blk in fields.items():
            i0, j0 = d.origin(rank)
            if blk.shape[0] != d.nxl:
                blk = blk[H : H + d.nxl, H : H + d.nyl, :]
            full[i0 : i0 + d.nxl, j0 : j0 + d.nyl, :] = blk
        return {r: full for r in self.local_ranks}

    def allreduce(self, vals, op="sum"):
        ts = torch.stack([torch.as_tensor(vals[r], dtype=torch.float64) for r in sorted(vals)])
        red = ts.sum() if op == "sum" else ts.max()
        return {r: red for r in self.local_ranks}

    def barrier(self):
        pass


class TorchDistComm:
    """One rank per process through torch.distributed. `fields` dicts carry
    exactly one entry: {my_rank: tensor}. Message buffers are staged through
    contiguous tensors on a staging device: CPU under gloo (works regardless
    of where the fields live), the fields' CUDA device under NCCL (buffers
    must be device-resident, and staying on-device is the point)."""

    def __init__(self, decomp, device="cpu"):
        import torch.distributed as dist

        self.dist = dist
        assert dist.is_initialized(), "call dist.init_process_group first"
        assert dist.get_world_size() == decomp.nranks
        self.decomp = decomp
        self.nranks = decomp.nranks
        self.rank = dist.get_rank()
        self.local_ranks = [self.rank]
        self.device = torch.device(device)
        backend = dist.get_backend()
        if backend == "nccl":
            assert torch.cuda.is_available(), "NCCL backend needs CUDA"
            # NCCL moves CUDA buffers only; fields may live anywhere (the
            # staging copy brings them to the device)
            self.stage_device = self.device if self.device.type == "cuda" else torch.device("cuda")
            # NCCL does not support point-to-point tags; matching relies on
            # posting order, which is deterministic here (symmetric pairs
            # posted in the same order on both sides)
            self.use_tags = False
        else:
            self.stage_device = torch.device("cpu")
            self.use_tags = True

    def _tag(self, tag):
        return tag if self.use_tags else 0

    def _sendrecv(self, ext, send_sl, recv_sl, dst, src, tag):
        """Send my `send_sl` slab to dst while receiving `recv_sl` from src."""
        if dst == self.rank and src == self.rank:
            ext[recv_sl] = ext[send_sl].clone()
            return
        send_buf = ext[send_sl].contiguous().to(self.stage_device)
        recv_buf = torch.empty_like(send_buf)
        ops = [
            self.dist.P2POp(self.dist.isend, send_buf, dst, tag=self._tag(tag)),
            self.dist.P2POp(self.dist.irecv, recv_buf, src, tag=self._tag(tag)),
        ]
        for req in self.dist.batch_isend_irecv(ops):
            req.wait()
        ext[recv_sl] = recv_buf.to(ext.device)

    def _exchange_dim(self, fields, dim, n_loc):
        d, H = self.decomp, self.decomp.H
        ext = fields[self.rank]
        nb = d.neighbors(self.rank)
        minus = nb["xm"] if dim == 0 else nb["ym"]
        plus = nb["xp"] if dim == 0 else nb["yp"]
        send_m, send_p, recv_m, recv_p = send_recv_slices(n_loc, H, dim)
        # pass 1: everyone sends trailing slab to plus, receives from minus
        self._sendrecv(ext, send_p, recv_m, plus, minus, tag=10 + dim)
        # pass 2: everyone sends leading slab to minus, receives from plus
        self._sendrecv(ext, send_m, recv_p, minus, plus, tag=20 + dim)

    def halo_exchange(self, fields):
        self._exchange_dim(fields, 0, self.decomp.nxl)
        self._exchange_dim(fields, 1, self.decomp.nyl)

    def pull_minus_edge(self, fields, dim):
        d = self.decomp
        n_loc = d.nxl if dim == 0 else d.nyl
        dst, src = edge_pull_slices(n_loc, d.H, dim)
        ext = fields[self.rank]
        nb = d.neighbors(self.rank)
        minus = nb["xm"] if dim == 0 else nb["ym"]
        plus = nb["xp"] if dim == 0 else nb["yp"]
        # my `src` slab goes to my PLUS neighbor's `dst`; I receive from minus
        self._sendrecv(ext, src, dst, plus, minus, tag=30 + dim)

    def scatter_nodes(self, nodes, comp):
        """Root (rank 0) broadcasts the full node array; every rank slices
        its own extended block. One-time seeding path, so the full-field
        broadcast cost is accepted for simplicity."""
        d = self.decomp
        nz_nodes = d.ext_shape(comp)[2]
        if self.rank == 0:
            assert nodes is not None, "scatter_nodes needs the full array on rank 0"
            buf = nodes.to(torch.float64).contiguous().to(self.stage_device)
            assert buf.shape == (d.nx, d.ny, nz_nodes), (
                f"node array for {comp!r} has shape {tuple(buf.shape)}, "
                f"expected {(d.nx, d.ny, nz_nodes)}"
            )
        else:
            buf = torch.empty(d.nx, d.ny, nz_nodes, dtype=torch.float64, device=self.stage_device)
        self.dist.broadcast(buf, src=0)
        full = d.scatter(buf, comp, fill_halos=True)
        return {self.rank: full[self.rank].to(self.device)}

    def allgather_nodes(self, fields):
        d, H = self.decomp, self.decomp.H
        ext = fields[self.rank]
        if ext.shape[0] != d.nxl:
            owned = ext[H : H + d.nxl, H : H + d.nyl, :].contiguous().to(self.stage_device)
        else:
            owned = ext.contiguous().to(self.stage_device)
        blocks = [torch.empty_like(owned) for _ in range(self.nranks)]
        self.dist.all_gather(blocks, owned)
        full = torch.zeros(d.nx, d.ny, ext.shape[2], dtype=ext.dtype, device=ext.device)
        for rank, blk in enumerate(blocks):
            i0, j0 = d.origin(rank)
            full[i0 : i0 + d.nxl, j0 : j0 + d.nyl, :] = blk.to(ext.device)
        return {self.rank: full}

    def allreduce(self, vals, op="sum"):
        t = torch.as_tensor(vals[self.rank], dtype=torch.float64).to(self.stage_device).clone()
        red_op = self.dist.ReduceOp.SUM if op == "sum" else self.dist.ReduceOp.MAX
        self.dist.all_reduce(t, op=red_op)
        return {self.rank: t}

    def barrier(self):
        self.dist.barrier()
