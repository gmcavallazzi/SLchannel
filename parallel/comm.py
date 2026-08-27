"""Communicator backends: emulated single-process ranks and torch.distributed.

Both expose the same call surface over `fields: dict[int, Tensor]` mapping
rank -> extended local array. EmulatedComm holds every rank's tensor in one
process (exchange = slice copies, deterministic, fast-suite friendly);
TorchDistComm holds exactly one rank per process and moves the identical
slabs through torch.distributed (gloo by default, slabs staged through
contiguous CPU tensors — correctness is the target on a single shared GPU;
NCCL needs CUDA MPS and is opt-in via SLC_DIST_BACKEND=nccl)."""

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
    exactly one entry: {my_rank: tensor}. Slabs are staged through contiguous
    CPU tensors so gloo works regardless of where the fields live."""

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

    def _sendrecv(self, ext, send_sl, recv_sl, dst, src, tag):
        """Send my `send_sl` slab to dst while receiving `recv_sl` from src."""
        if dst == self.rank and src == self.rank:
            ext[recv_sl] = ext[send_sl].clone()
            return
        send_buf = ext[send_sl].contiguous().cpu()
        recv_buf = torch.empty_like(send_buf)
        ops = [
            self.dist.P2POp(self.dist.isend, send_buf, dst, tag=tag),
            self.dist.P2POp(self.dist.irecv, recv_buf, src, tag=tag),
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

    def allgather_nodes(self, fields):
        d, H = self.decomp, self.decomp.H
        ext = fields[self.rank]
        if ext.shape[0] != d.nxl:
            owned = ext[H : H + d.nxl, H : H + d.nyl, :].contiguous().cpu()
        else:
            owned = ext.contiguous().cpu()
        blocks = [torch.empty_like(owned) for _ in range(self.nranks)]
        self.dist.all_gather(blocks, owned)
        full = torch.zeros(d.nx, d.ny, ext.shape[2], dtype=ext.dtype, device=ext.device)
        for rank, blk in enumerate(blocks):
            i0, j0 = d.origin(rank)
            full[i0 : i0 + d.nxl, j0 : j0 + d.nyl, :] = blk.to(ext.device)
        return {self.rank: full}

    def allreduce(self, vals, op="sum"):
        t = torch.as_tensor(vals[self.rank], dtype=torch.float64).cpu().clone()
        red_op = self.dist.ReduceOp.SUM if op == "sum" else self.dist.ReduceOp.MAX
        self.dist.all_reduce(t, op=red_op)
        return {self.rank: t}

    def barrier(self):
        self.dist.barrier()
