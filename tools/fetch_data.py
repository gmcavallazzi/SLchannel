"""Fetch the datasets slChannel validates against.

    python tools/fetch_data.py mkm180      # reference DNS profiles (~1 MB)
    python tools/fetch_data.py --list
    python tools/fetch_data.py --all

Nothing here is redistributed with the source: the reference DNS is downloaded
from the Oden Institute (UT Austin), and the restart seed from the project's
Zenodo record. Both are checksummed after download.

AXIS CONVENTION -- the reason this script exists rather than a raw download.
The published channel data uses y as the WALL-NORMAL direction and z as
spanwise; slChannel uses z wall-normal and y spanwise:

    reference:   x streamwise (u),  y wall-normal (v),  z spanwise (w)
    slChannel:   x streamwise (u),  y spanwise    (v),  z wall-normal (w)

so the Reynolds stresses must be REMAPPED, not merely renamed:

    R_uu -> uu       R_vv (wall-normal) -> ww
    R_uv -> uw       R_ww (spanwise)    -> vv

Overlaying the reference columns in file order silently swaps v'v' and w'w'.
The CSV written here is already remapped and uses slChannel's naming.

The published profiles are CLOSED-channel data spanning y/delta = 0..1 (the
lower half, by symmetry).
"""

import argparse
import hashlib
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "data")

MKM_BASE = "https://turbulence.oden.utexas.edu/data/MKM"
LM_BASE = "https://turbulence.oden.utexas.edu/channel2015/data"

DATASETS = {
    "lm1000": dict(
        kind="lm",
        retau=1000,
        dest="reference/lm1000",
        cite=("M. Lee & R. D. Moser, J. Fluid Mech. 774, 395 (2015)."),
        note="Reference DNS profiles at Re_tau = 1000.5 (8pi x 3pi box).",
    ),
    "lm2000": dict(
        kind="lm",
        retau=2000,
        dest="reference/lm2000",
        cite=("M. Lee & R. D. Moser, J. Fluid Mech. 774, 395 (2015)."),
        note="Reference DNS profiles at Re_tau = 1994.8 (8pi x 3pi box).",
    ),
    "mkm180": dict(
        kind="mkm",
        retau=180,
        dest="reference/mkm180.csv",
        cite=(
            "R. D. Moser, J. Kim & N. N. Mansour, Phys. Fluids 11, 943 (1999); "
            "J. Kim, P. Moin & R. Moser, JFM 177, 133 (1987)."
        ),
        note="Reference DNS profiles at Re_tau = 178.",
    ),
    "m950_seed": dict(
        kind="zenodo",
        # URL filled in when the Zenodo version with this file is published;
        # until then place the file at data/m950_seed_768x640x320.npz by hand
        # (the sha256 below verifies it either way).
        url=None,
        sha256="63039e8189fad3fc349782073b9f24d555abbd1fdab8ecc3d720d5dd09514df9",
        dest="m950_seed_768x640x320.npz",
        size_mb=4851,
        cite=(
            "G. M. Cavallazzi, seed fields for slChannel validation, Zenodo, "
            "CC-BY-4.0. https://doi.org/10.5281/zenodo.22099568"
        ),
        note="Converged M950-replica field (Re_tau = 912, 2pi x pi, t = 150.8).",
    ),
    "kmm180_seed": dict(
        kind="zenodo",
        # Pinned to the VERSION DOI, not the concept DOI: reproducing a result
        # needs these exact bytes, not "whatever is latest".
        url="https://zenodo.org/records/22099569/files/tor180c256_seed.npz?download=1",
        sha256="9630c6fec810920aaf838486dfdbf1ab39be8b9437a6ae341305be4603e769f8",
        dest="kmm180_seed_256cubed.npz",
        size_mb=548,
        cite=(
            "G. M. Cavallazzi, 'seed field SLchannel Re_tau = 180', Zenodo, "
            "CC-BY-4.0. https://doi.org/10.5281/zenodo.22099568"
        ),
        note="256^3 restart field for the headline KMM validation case.",
    ),
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    print(f"  GET {url}")
    with urllib.request.urlopen(url, timeout=300) as r, open(part, "wb") as fh:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            fh.write(block)
            got += len(block)
            if total:
                print(
                    f"\r  {100 * got // total:3d}%  {got >> 20} / {total >> 20} MiB",
                    end="",
                    flush=True,
                )
        if total:
            print()
    os.replace(part, dest)
    return dest


def fetch_mkm(spec):
    """Download the MKM profiles and write the axis-remapped CSV."""
    retau = spec["retau"]
    base = f"{MKM_BASE}/chan{retau}/profiles"
    tmp = os.path.join(DATA, "_mkm_raw")
    os.makedirs(tmp, exist_ok=True)

    means = download(f"{base}/chan{retau}.means", os.path.join(tmp, "means"))
    stress = download(f"{base}/chan{retau}.reystress", os.path.join(tmp, "reystress"))

    def read(path):
        rows = []
        for line in open(path):
            line = line.strip()
            if not line or line[0] in "#%":
                continue
            try:
                rows.append([float(v) for v in line.split()])
            except ValueError:
                continue  # column-name banner
        return rows

    means_rows, stress_rows = read(means), read(stress)
    if not means_rows or not stress_rows:
        sys.exit("could not parse the downloaded MKM profiles")

    # chan.means:     y  y+  Umean  dUmean/dy  Wmean  dWmean/dy  Pmean
    # chan.reystress: y  y+  R_uu  R_vv  R_ww  R_uv  R_uw  R_vw
    out = os.path.join(DATA, spec["dest"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write(f"# MKM Re_tau={retau} profiles, downloaded from {base}\n")
        fh.write(f"# {spec['cite']}\n")
        fh.write("# Axes remapped to slChannel's convention: reference y (wall-normal)\n")
        fh.write("#   -> z, reference R_vv -> ww, R_ww -> vv, R_uv -> uw.\n")
        fh.write("z_delta,z_plus,U_plus,uu_plus,vv_plus,ww_plus,uw_plus\n")
        for m, s in zip(means_rows, stress_rows):
            y, yplus, umean = m[0], m[1], m[2]
            r_uu, r_vv, r_ww, r_uv = s[2], s[3], s[4], s[5]
            #                     wall-normal -> ww,  spanwise -> vv
            fh.write(
                f"{y:.6e},{yplus:.6e},{umean:.6e},{r_uu:.6e},{r_ww:.6e},{r_vv:.6e},{r_uv:.6e}\n"
            )

    for f in (means, stress):
        os.remove(f)
    os.rmdir(tmp)
    print(f"  wrote {out} ({len(means_rows)} points)")
    return out


def fetch_lm(spec):
    """Download the Lee & Moser (2015) profiles and write axis-remapped CSVs.

    The mean and fluctuation files are tabulated on DIFFERENT wall-normal
    grids, so they are written as two CSVs rather than zipped into one.
    LM's y is wall-normal and z spanwise, so as with MKM the stresses are
    remapped: their v'v' (wall-normal) -> ww, w'w' (spanwise) -> vv,
    u'v' -> uw. Everything is already in + units in the source files.
    """
    retau = spec["retau"]
    tmp = os.path.join(DATA, "_lm_raw")
    os.makedirs(tmp, exist_ok=True)

    mean = download(f"{LM_BASE}/LM_Channel_{retau}_mean_prof.dat", os.path.join(tmp, "mean"))
    fluc = download(f"{LM_BASE}/LM_Channel_{retau}_vel_fluc_prof.dat", os.path.join(tmp, "fluc"))

    def read(path):
        rows = []
        for line in open(path):
            line = line.strip()
            if not line or line[0] in "#%":
                continue
            try:
                rows.append([float(v) for v in line.split()])
            except ValueError:
                continue
        return rows

    mean_rows, fluc_rows = read(mean), read(fluc)
    if not mean_rows or not fluc_rows:
        sys.exit("could not parse the downloaded Lee-Moser profiles")

    out_base = os.path.join(DATA, spec["dest"])
    os.makedirs(os.path.dirname(out_base), exist_ok=True)

    # LM mean: y/delta  y+  U+  dU+/dy  W+  P
    out = out_base + "_mean.csv"
    with open(out, "w") as fh:
        fh.write(f"# Lee-Moser Re_tau={retau} mean profile, from {LM_BASE}\n")
        fh.write(f"# {spec['cite']}\n")
        fh.write("# Reference y (wall-normal) -> slChannel z.\n")
        fh.write("z_delta,z_plus,U_plus\n")
        for r in mean_rows:
            fh.write(f"{r[0]:.6e},{r[1]:.6e},{r[2]:.6e}\n")
    print(f"  wrote {out} ({len(mean_rows)} points)")

    # LM fluc: y/delta  y+  u'u'  v'v'  w'w'  u'v'  u'w'  v'w'  k  (all +)
    out = out_base + "_fluc.csv"
    with open(out, "w") as fh:
        fh.write(f"# Lee-Moser Re_tau={retau} velocity fluctuations, from {LM_BASE}\n")
        fh.write(f"# {spec['cite']}\n")
        fh.write("# Axes remapped to slChannel's convention: reference y (wall-normal)\n")
        fh.write("#   -> z, reference v'v' -> ww, w'w' -> vv, u'v' -> uw.\n")
        fh.write("z_delta,z_plus,uu_plus,vv_plus,ww_plus,uw_plus\n")
        for r in fluc_rows:
            uu, vv_wn, ww_sp, uv = r[2], r[3], r[4], r[5]
            fh.write(f"{r[0]:.6e},{r[1]:.6e},{uu:.6e},{ww_sp:.6e},{vv_wn:.6e},{uv:.6e}\n")
    print(f"  wrote {out} ({len(fluc_rows)} points)")

    for f in (mean, fluc):
        os.remove(f)
    os.rmdir(tmp)
    return out_base


def fetch_zenodo(name, spec):
    dest = os.path.join(DATA, spec["dest"])
    if spec.get("sha256") and os.path.exists(dest) and sha256(dest) == spec["sha256"]:
        print(f"  {dest} already present and verified")
        return dest
    if not spec.get("url"):
        sys.exit(
            f"'{name}' is not downloadable yet: its Zenodo version has not\n"
            f"been published. Place the file at {dest} by hand and rerun to\n"
            f"verify it against the recorded checksum."
        )
    download(spec["url"], dest)
    got = sha256(dest)
    if got != spec["sha256"]:
        sys.exit(f"checksum mismatch for {dest}\n  expected {spec['sha256']}\n  got      {got}")
    print(f"  verified sha256 {got}")
    return dest


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("names", nargs="*", help="datasets to fetch")
    ap.add_argument("--list", action="store_true", help="list what is available")
    ap.add_argument("--all", action="store_true", help="fetch everything available")
    args = ap.parse_args()

    if args.list or (not args.names and not args.all):
        print("available datasets:\n")
        for name, spec in DATASETS.items():
            size = f"  (~{spec['size_mb']} MB)" if spec.get("size_mb") else ""
            print(f"  {name:14s} {spec['note']}{size}")
            print(f"  {'':14s} cite: {spec['cite']}\n")
        return 0

    names = list(DATASETS) if args.all else args.names
    for name in names:
        if name not in DATASETS:
            sys.exit(f"unknown dataset {name!r}; try --list")
        spec = DATASETS[name]
        print(f"{name}: {spec['note']}")
        if spec["kind"] == "mkm":
            fetch_mkm(spec)
        elif spec["kind"] == "lm":
            fetch_lm(spec)
        else:
            fetch_zenodo(name, spec)
        print(f"  cite: {spec['cite']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
