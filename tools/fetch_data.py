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

DATASETS = {
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


def fetch_zenodo(name, spec):
    if not spec.get("url"):
        sys.exit(
            f"'{name}' is not downloadable yet: the Zenodo data record has not\n"
            f"been published, so no URL or checksum is recorded. See\n"
            f"docs/REPRODUCING.md for the current status and for how to\n"
            f"regenerate this field locally."
        )
    dest = os.path.join(DATA, spec["dest"])
    if os.path.exists(dest) and sha256(dest) == spec["sha256"]:
        print(f"  {dest} already present and verified")
        return dest
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
        else:
            fetch_zenodo(name, spec)
        print(f"  cite: {spec['cite']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
