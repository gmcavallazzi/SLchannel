"""Re180 replication runs for the 1-GPU vs 4-rank comparison.

Both modes run the SAME protocol -- the KMM triquintic 256^3 closed channel,
seeded by interpolating the converged CaNS checkpoint, fixed production dt,
statistics accumulated by the production TurbulenceStats at the production
cadence -- so the results must match bitwise (the decomposed step is
bit-identical to the monolithic one, including the gathered bulk-forcing
reduction):

    python parallel/run_re180_4rank.py --mode 4rank --out results/re180_4rank
    python parallel/run_re180_4rank.py --mode mono  --out results/re180_mono
    python parallel/compare_re180.py   results/re180_4rank results/re180_mono

Pause: touch <out>/STOP (checkpoint written, clean exit; no restart -- this
is a single fixed-window replication run).
"""

import argparse
import os
import sys
import time

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from parallel.comm import EmulatedComm  # noqa: E402
from parallel.decomp import Decomp, nodes_to_mono  # noqa: E402
from parallel.sl_local import required_halo  # noqa: E402
from parallel.step import DecomposedBDF2  # noqa: E402
from slchannel.solver import SLChannelFlow  # noqa: E402
from slchannel.utils import compute_u_tau  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["4rank", "mono"], default="4rank")
    ap.add_argument("--config", default="configs/kmm180_sl_bdf2_quintic.yaml")
    ap.add_argument("--seed", default="data/cans_re180tc_256x256x256.npz")
    ap.add_argument("--out", default="results/re180_4rank")
    ap.add_argument("--t-total", type=float, default=155.0)
    ap.add_argument("--t-stats", type=float, default=5.0)
    ap.add_argument("--px", type=int, default=2)
    ap.add_argument("--py", type=int, default=2)
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(open(args.config))
    cfg["output"]["results_folder"] = args.out
    cfg["statistics"]["t_stats"] = args.t_stats
    cfg["initialization"] = {
        "type": "interpolate",
        "field_file": args.seed,
        "reset_time": True,
    }
    os.makedirs(args.out, exist_ok=True)
    tmp_cfg = os.path.join(args.out, "_run_config.yaml")
    yaml.safe_dump(cfg, open(tmp_cfg, "w"))

    mono = SLChannelFlow(config_file=tmp_cfg)
    dt = cfg["time"]["dt"]
    nx, ny = mono.nx, mono.ny
    ts = mono.turbulence_stats
    n_stats, n_out = mono.n_stats, mono.n_out
    n_steps = int(round(args.t_total / dt))
    stop_path = os.path.join(args.out, "STOP")

    dec = None
    if args.mode == "4rank":
        umax = float(
            torch.stack([mono.u.abs().max(), mono.v.abs().max(), mono.w.abs().max()]).max()
        )
        disp = dt * umax / min(mono.dx, mono.dy)
        H = required_halo(mono.sl.order, disp_cells=disp)
        d = Decomp(args.px, args.py, nx, ny, mono.nz, H=H)
        comm = EmulatedComm(d, device=mono.device)
        dec = DecomposedBDF2(mono, d, comm, poisson="pencil", use_triton=True)
        print(
            f"[4rank] {args.px}x{args.py} ranks, H={H} (disp {disp:.2f} cells/dt, "
            f"order {mono.sl.order}), pencil Poisson, Triton-local SL",
            flush=True,
        )
    else:
        print("[mono] production single-device step_sl_bdf2 path", flush=True)

    def current_fields():
        if dec is None:
            return mono.u, mono.v, mono.w
        return (
            nodes_to_mono(dec.gather_nodes("u"), "u", nx, ny),
            nodes_to_mono(dec.gather_nodes("v"), "v", nx, ny),
            nodes_to_mono(dec.gather_nodes("w"), "w", nx, ny),
        )

    t = 0.0
    t0 = time.time()
    print(f"{'Step':>6} {'Time':>10} {'u_tau':>10} {'wall s/step':>12}", flush=True)
    for step in range(1, n_steps + 1):
        if dec is None:
            mono.step_sl_bdf2(dt)
        else:
            dec.step(dt)
        t += dt
        if ts is not None and t >= args.t_stats and step % n_stats == 0:
            u, v, w = current_fields()
            u_tau = compute_u_tau(u, mono.z_c, mono.nu, top_wall_bc_type=mono.top_wall_bc_type)
            ts.accumulate_statistics(u, v, w, u_tau)
        if step % n_out == 0:
            u, _, _ = current_fields()
            u_tau = float(
                compute_u_tau(u, mono.z_c, mono.nu, top_wall_bc_type=mono.top_wall_bc_type)
            )
            print(
                f"{step:6d} {t:10.4f} {u_tau:10.6f} {(time.time() - t0) / step:12.3f}",
                flush=True,
            )
        if step % 2000 == 0 or os.path.exists(stop_path):
            u, v, w = current_fields()
            torch.save(
                {"u": u.cpu(), "v": v.cpu(), "w": w.cpu(), "time": t, "step": step},
                os.path.join(args.out, "checkpoint.pt"),
            )
            if ts is not None and ts.n_samples > 0:
                ts.save_state(os.path.join(args.out, "turbulence_stats_state.npz"))
            if os.path.exists(stop_path):
                print(f"STOP file found at step {step}, t = {t:.4f} -- exiting", flush=True)
                break

    if ts is not None and ts.n_samples > 0:
        ts.save_statistics(os.path.join(args.out, "turbulence_stats.npz"))
        print(f"statistics over {ts.n_samples} samples saved", flush=True)
    u, v, w = current_fields()
    torch.save(
        {"u": u.cpu(), "v": v.cpu(), "w": w.cpu(), "time": t, "step": step},
        os.path.join(args.out, "fields_final.pt"),
    )
    print(
        f"done ({args.mode}): {step} steps, t = {t:.4f}, {(time.time() - t0) / 60:.1f} min total",
        flush=True,
    )


if __name__ == "__main__":
    main()
