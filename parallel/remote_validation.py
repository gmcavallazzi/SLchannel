"""Self-contained multi-GPU validation ladder for a remote node.

Purpose: validate the decomposed solver's REAL multi-GPU path (NCCL, one
rank per GPU) on a machine with >= 2 CUDA devices — the one thing that
cannot be tested on a single shared GPU. No slurm, no input data, no
network access needed beyond the initial clone/install.

On the remote node, from the repository root:

    pip install -e .
    python parallel/remote_validation.py            # full ladder, ~10-20 min
    python parallel/remote_validation.py --quick    # small grids only, ~5 min

Everything lands in remote_validation/: per-stage logs and the single file
to send back, remote_validation/report.json. On failure, also send the
matching *.log files.

Stages (each guarded; a failure is recorded and the ladder continues):
    env             machine / torch / GPU inventory
    suite           pytest parallel/tests (emulated backend + GPU Triton)
    dist_gloo       4-process gloo comm + step checks (CPU fields)
    dist_nccl       same checks over real NCCL, one GPU per rank
    prod_bitwise    monolithic GPU run vs 2-rank NCCL production run,
                    gathered Poisson + gathered bulk: fields must match
                    bitwise (identical GPUs, identical reduction order)
    prod_pencil     2-rank NCCL production run, pencil Poisson + local
                    bulk (the real production configuration): completes,
                    stays close to the monolithic trajectory
    perf            s/step: monolithic 1 GPU vs 2-rank NCCL (pencil+local
                    and gathered+local) at 256^3   [skipped with --quick]
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

OUT = os.path.join(REPO, "remote_validation")
REPORT = {"env": {}, "stages": {}, "overall": "PASS"}
# hidden rehearsal knobs (leave unset on the real node): run the "NCCL"
# stages over another backend / device, e.g. SLC_RV_BACKEND=gloo
# SLC_RV_DEVICE=cpu rehearses the full ladder on a busy single-GPU machine
RV_BACKEND = os.environ.get("SLC_RV_BACKEND", "nccl")
RV_DEVICE = os.environ.get("SLC_RV_DEVICE", "cuda")


def _log_path(stage):
    return os.path.join(OUT, f"{stage}.log")


def _record(stage, status, **details):
    REPORT["stages"][stage] = {"status": status, **details}
    if status == "fail":
        REPORT["overall"] = "FAIL"
    print(f"[{stage}] {status.upper()}  {details}", flush=True)


def _run(cmd, stage, env=None, timeout=3600):
    """Run a subprocess, tee output to the stage log, return (rc, text)."""
    e = dict(os.environ)
    if env:
        e.update(env)
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd,
            cwd=REPO,
            env=e,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        rc, text = p.returncode, p.stdout
    except subprocess.TimeoutExpired as ex:
        rc, text = -9, (ex.stdout or "") + "\n[TIMEOUT]"
    with open(_log_path(stage), "a") as fh:
        fh.write(f"$ {' '.join(cmd)}\n{text}\n")
    return rc, text, time.time() - t0


def _write_cfg(name, nx, ny, nz, n_steps, folder, dt=0.01):
    cfg = {
        "grid": {"nx": nx, "ny": ny, "nz": nz},
        "domain": {"Lx": 6.283185307179586, "Ly": 3.141592653589793, "Lz": 2.0},
        "flow": {"Re": 2800.0, "Re_tau": 180.0, "U_bulk": 1.0, "gamma": 1.8},
        "initialization": {"type": "vortices", "n_vortices": 2},
        "boundary_conditions": {"top_wall": {"type": "dirichlet"}},
        "solver": {"type": "fft"},
        "advection": {"scheme": "sl"},
        "sl": {"interp_order": 6, "n_traj_iters": 2, "interp_dtype": "fp32_accum64"},
        "compute": {"device": RV_DEVICE},
        "time": {
            "dt": dt,
            "n_steps": n_steps,
            "t_max": 1e9,
            "CFL_target": 3.0,
            "dt_update_interval": 0,
            "dt_max": dt,
            "dt_min": dt / 100,
            "scheme": "IMEX",
            "stop_on_blowup": False,
        },
        "output": {
            "results_folder": os.path.join(OUT, folder),
            "n_out": max(1, n_steps // 3),
            "n_save": 10**8,
            "t_snapshot": 0.0,
        },
        "statistics": {"n_stats": 0},
    }
    path = os.path.join(OUT, name)
    yaml.safe_dump(cfg, open(path, "w"))
    return path


def _mono_run(cfg_path, n_steps):
    """Monolithic GPU run in-process; returns (fields dict, s/step over the
    post-bootstrap steps)."""
    from slchannel.solver import SLChannelFlow

    mono = SLChannelFlow(config_file=cfg_path)
    sync = torch.cuda.synchronize if mono.device.type == "cuda" else (lambda: None)
    for _ in range(3):
        mono.step_sl_bdf2(mono.dt)
    sync()
    t0 = time.time()
    for _ in range(n_steps - 3):
        mono.step_sl_bdf2(mono.dt)
    sync()
    sps = (time.time() - t0) / (n_steps - 3)
    fields = {c: getattr(mono, c).cpu().numpy() for c in "uvw"}
    del mono
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return fields, sps


def _dist_run(cfg_path, stage, poisson, bulk, timeout=3600):
    """2-rank NCCL production run via torchrun; returns (rc, s/step or None,
    final-fields npz path)."""
    cfg = yaml.safe_load(open(cfg_path))
    folder = cfg["output"]["results_folder"]
    rc, text, wall = _run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "-m",
            "parallel.production",
            cfg_path,
            "--px",
            "2",
            "--py",
            "1",
            "--backend",
            "dist",
            "--poisson",
            poisson,
            "--bulk",
            bulk,
        ],
        stage,
        env={"SLC_DIST_BACKEND": RV_BACKEND},
        timeout=timeout,
    )
    # the driver prints a cumulative "[x.xxx s/step]" at each n_out; combine
    # the first and last to strip the bootstrap from the average
    sps = None
    brackets = [
        (int(s), float(v))
        for s, v in re.findall(r"^\s*(\d+) .*\[(\d+\.\d+) s/step\]", text, re.MULTILINE)
    ]
    if len(brackets) >= 2:
        (s1, a1), (s2, a2) = brackets[0], brackets[-1]
        if s2 > s1:
            sps = (s2 * a2 - s1 * a1) / (s2 - s1)
    elif brackets:
        sps = brackets[0][1]
    return rc, sps, os.path.join(folder, "fields_final.npz")


def _compare(final_npz, mono_fields):
    d = np.load(final_npz)
    return {c: float(np.abs(d[c] - mono_fields[c]).max()) for c in "uvw"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small grids, skip perf")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    REPORT["env"] = {
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": ".".join(map(str, torch.cuda.nccl.version())) if ngpu else None,
        "gpus": [torch.cuda.get_device_name(i) for i in range(ngpu)],
        "quick": args.quick,
    }
    print(f"env: {REPORT['env']}", flush=True)
    nccl_ok = ngpu >= 2 if RV_BACKEND == "nccl" else True

    # -- suite ---------------------------------------------------------------
    rc, text, wall = _run([sys.executable, "-m", "pytest", "parallel/tests", "-q"], "suite")
    tail = text.strip().splitlines()[-1] if text.strip() else ""
    _record("suite", "pass" if rc == 0 else "fail", tail=tail, seconds=round(wall))

    # -- dist_gloo / dist_nccl ----------------------------------------------
    for stage, backend, world, enabled in (
        ("dist_gloo", "gloo", 4, True),
        ("dist_nccl", RV_BACKEND, min(2, ngpu) if RV_BACKEND == "nccl" else 2, nccl_ok),
    ):
        if not enabled:
            _record(stage, "skip", reason=f"needs >= 2 GPUs, found {ngpu}")
            continue
        jd = os.path.join(OUT, f"{stage}_json")
        rc, text, wall = _run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={world}",
                "parallel/run_dist_test.py",
            ],
            stage,
            env={"SLC_DIST_BACKEND": backend, "SLC_DIST_JSON_DIR": jd},
        )
        worst = {}
        try:
            for r in range(world):
                res = json.load(open(os.path.join(jd, f"rank{r}.json")))
                for k, v in res.items():
                    worst[k] = max(worst.get(k, 0.0), abs(v))
            ok = rc == 0 and all(v <= 1e-12 for v in worst.values())
        except FileNotFoundError:
            ok, worst = False, {"error": "rank json missing (see log)"}
        _record(stage, "pass" if ok else "fail", worst=worst, seconds=round(wall))

    # -- production runs vs monolithic --------------------------------------
    n_steps = 20
    gsz = (96, 96, 64) if args.quick else (128, 128, 96)
    if not nccl_ok:
        _record("prod_bitwise", "skip", reason="needs >= 2 GPUs")
        _record("prod_pencil", "skip", reason="needs >= 2 GPUs")
    else:
        cfg_m = _write_cfg("mono.yaml", *gsz, n_steps, "out_mono")
        mono_fields, mono_sps = _mono_run(cfg_m, n_steps)

        cfg_b = _write_cfg("bitwise.yaml", *gsz, n_steps, "out_bitwise")
        rc, _, final = _dist_run(cfg_b, "prod_bitwise", "gathered", "gathered")
        try:
            errs = _compare(final, mono_fields)
            ok = rc == 0 and max(errs.values()) <= 1e-12
        except FileNotFoundError:
            ok, errs = False, {"error": "fields_final.npz missing (see log)"}
        _record("prod_bitwise", "pass" if ok else "fail", max_diff=errs, n_steps=n_steps)

        cfg_p = _write_cfg("pencil.yaml", *gsz, n_steps, "out_pencil")
        rc, _, final = _dist_run(cfg_p, "prod_pencil", "pencil", "local")
        try:
            errs = _compare(final, mono_fields)
            # pencil FFT differs in the last bit; short horizon stays tiny
            ok = rc == 0 and max(errs.values()) <= 1e-8
        except FileNotFoundError:
            ok, errs = False, {"error": "fields_final.npz missing (see log)"}
        _record("prod_pencil", "pass" if ok else "fail", max_diff=errs, n_steps=n_steps)

    # -- perf ----------------------------------------------------------------
    if args.quick or not nccl_ok:
        _record("perf", "skip", reason="--quick" if args.quick else "needs >= 2 GPUs")
    else:
        n_perf = 30
        cfg = _write_cfg("perf_mono.yaml", 256, 256, 256, n_perf, "out_perf_mono", dt=0.005)
        _, mono_sps = _mono_run(cfg, n_perf)
        timings = {"mono_1gpu": round(mono_sps, 4)}
        for tag, poisson, bulk in (
            ("nccl_2gpu_pencil_local", "pencil", "local"),
            ("nccl_2gpu_gathered_local", "gathered", "local"),
        ):
            cfgd = _write_cfg(
                f"perf_{tag}.yaml", 256, 256, 256, n_perf, f"out_perf_{tag}", dt=0.005
            )
            rc, sps, _ = _dist_run(cfgd, "perf", poisson, bulk)
            timings[tag] = round(sps, 4) if (rc == 0 and sps) else f"failed rc={rc}"
        ok = all(isinstance(v, float) for v in timings.values())
        _record(
            "perf",
            "pass" if ok else "fail",
            s_per_step=timings,
            grid="256^3",
            note="dist numbers strip the bootstrap steps from the average",
        )

    with open(os.path.join(OUT, "report.json"), "w") as fh:
        json.dump(REPORT, fh, indent=2)
    print(f"\n=== {REPORT['overall']} ===")
    print(f"report: {os.path.join(OUT, 'report.json')}  (send this file back)")
    return 0 if REPORT["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
