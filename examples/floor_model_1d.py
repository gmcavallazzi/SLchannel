"""1-D model of the SL spectral-floor mechanism (2026-08-03).

Passive scalar advected on a periodic 1-D grid by c(x,t) = c0 + zeta(x,t),
where zeta is an AR(1)-in-time random field (correlation time tau) — the
model of turbulent sweeping that decorrelates the trajectory velocity
between steps. Every scheme sees the SAME velocity history; they differ
only in how they build the advecting velocity and combine remaps:

  ab2  : c* = 1.5 c^n - 0.5 c^{n-1}          (v2 analog: extrapolation)
  none : c* = c^n                             (no extrapolation, O(dt))
  pc   : c* = 0.5 (c^n + c^{n+1})             (ideal projected predictor)
  bdf2 : U* = 2 c^n - c^{n-1}; u^{n+1} = (4 ubar - ubarbar)/3
         with feet at dt and 2dt (Boukir)

The exact advecting velocity over [t^n, t^{n+1}] is the midpoint value
c^{n+1/2}; each scheme's departure-point error is (c* - c^{n+1/2}) dt.
A fixed low-k band is restored every step (energy injection); the metric
is the equilibrium tail energy relative to the forced-band energy (the
noise-free tail decays to round-off, so ratios to it are meaningless).
Cubic Lagrange remap, matching the solver's order=4. A per-step viscous
factor exp(-nu_model k^2 dt) matched to the channel tail (nu kx^2 dt ~
0.06 at the last resolved kx for dt+ = 0.4) stands in for the omitted
diffusion.

The channel dictionary: dt/tau ~ dt+/0.22 (floor threshold at CFL_x~0.65),
zeta_rms/c0 ~ u'/U at z+~15 ~ 0.25.

Prediction to check against the GPU sweep: the (4/3, -1/3) BDF2 foot
combination amplifies phase-decorrelated tail content by up to 17/9 per
step — more than cubic-interpolation damping (~0.9) plus weak viscosity
(~0.94) can remove — so bdf2's tail should GROW at dt/tau >~ 1, while all
single-foot schemes (ab2/none/pc) stay bounded with floors ordered
ab2 > none > pc.

Usage: python examples/floor_model_1d.py   (CPU, seconds)
Output: figures/model_floor_1d.png
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "mathtext.fontset": "stix", "font.family": "STIXGeneral"})

rng = np.random.default_rng(7)

N = 256
H = 1.0
X = np.arange(N) * H
C0 = 1.0
CFL_FRAC = 0.65  # c0*dt/h at the channel's floor threshold
ZETA_REL = 0.25  # zeta_rms / c0
K_FORCE = 6  # restored band: |k| <= K_FORCE
N_STEPS = 4000
N_AVG = 2000  # spectrum average window (after spin-up)
DT_OVER_TAU = [0.5, 1.0, 2.0]

k = np.fft.rfftfreq(N, d=H) * 2 * np.pi
K_TAIL = slice(len(k) - 26, len(k) - 1)  # last 25 modes below Nyquist
# per-step viscous factor, matched to nu*kx^2*dt ~ 0.06 at the channel's
# last resolved kx (dt+ = 0.4): exp(-0.06 (k/kmax)^2)
VISC = np.exp(-0.06 * (k / k[-1]) ** 2)


def cubic_remap(u, xd):
    """Periodic cubic-Lagrange interpolation of u at points xd (per point)."""
    s = xd / H
    j = np.floor(s).astype(int)
    a = s - j  # in [0,1)
    im1, i0, i1, i2 = ((j - 1) % N, j % N, (j + 1) % N, (j + 2) % N)
    w_m1 = -a * (a - 1) * (a - 2) / 6.0
    w_0 = (a + 1) * (a - 1) * (a - 2) / 2.0
    w_1 = -(a + 1) * a * (a - 2) / 2.0
    w_2 = (a + 1) * a * (a - 1) / 6.0
    return w_m1 * u[im1] + w_0 * u[i0] + w_1 * u[i1] + w_2 * u[i2]


def make_zeta_sequence(n_steps, rho):
    """AR(1) random fields: zeta^{n+1} = rho*zeta^n + sqrt(1-rho^2)*xi.
    Broadband white-in-x, rms ZETA_REL*C0. Also returns midpoint values
    (independent draws consistent with AR(1) bridging is overkill; use the
    average of endpoints as the 'true' mid velocity — the schemes are being
    compared, not absolutely calibrated)."""
    z = np.empty((n_steps + 2, N))
    z[0] = rng.normal(0.0, ZETA_REL * C0, N)
    q = np.sqrt(1.0 - rho * rho)
    for n in range(1, n_steps + 2):
        z[n] = rho * z[n - 1] + q * rng.normal(0.0, ZETA_REL * C0, N)
    return z


def spectrum(u):
    return np.abs(np.fft.rfft(u)) ** 2 / N**2


def run_scheme(scheme, zeta, dt, u0, force_hat, n_steps):
    """March n_steps; return time-averaged spectrum over the last N_AVG."""
    u_nm1 = u0.copy()
    u = u0.copy()
    acc = np.zeros(len(k))
    n_acc = 0
    for n in range(1, n_steps + 1):
        c_n = C0 + zeta[n]
        c_nm1 = C0 + zeta[n - 1]
        c_mid = C0 + 0.5 * (zeta[n] + zeta[n + 1])  # 'true' mid velocity
        if scheme == "ab2":
            cs = 1.5 * c_n - 0.5 * c_nm1
        elif scheme == "none":
            cs = c_n
        elif scheme == "pc":
            cs = c_mid
        elif scheme == "bdf2":
            cs = 2.0 * c_n - c_nm1
        if scheme == "bdf2":
            ub = cubic_remap(u, X - cs * dt)
            ubb = cubic_remap(u_nm1, X - 2.0 * cs * dt)
            u_new = (4.0 * ub - ubb) / 3.0
        else:
            u_new = cubic_remap(u, X - cs * dt)
        u_nm1, u = u, u_new
        # viscous damping + restore the forced band (energy injection)
        uh = np.fft.rfft(u) * VISC
        uh[: K_FORCE + 1] = force_hat[: K_FORCE + 1]
        u = np.fft.irfft(uh, n=N)
        if not np.all(np.isfinite(u)) or np.abs(u).max() > 1e12:
            return None  # blew up
        if n > n_steps - N_AVG:
            acc += spectrum(u)
            n_acc += 1
    return acc / n_acc


def main():
    dt = CFL_FRAC * H / C0
    u0 = np.real(
        np.fft.irfft(
            np.where(
                np.arange(len(k)) <= K_FORCE,
                N * rng.normal(size=len(k)) * np.exp(1j * rng.uniform(0, 2 * np.pi, len(k))),
                0.0,
            ),
            n=N,
        )
    )
    u0 /= np.std(u0)
    force_hat = np.fft.rfft(u0)

    fig, axes = plt.subplots(1, len(DT_OVER_TAU), figsize=(12.5, 4.0), sharey=True)
    print(f"CFL={CFL_FRAC}, zeta_rms/c0={ZETA_REL}, N={N}, tail = last 25 modes")
    print(
        f"{'dt/tau':>7} {'ab2':>10} {'none':>10} {'pc':>10} {'bdf2':>10}   (tail energy / forced-band energy)"
    )
    for ax, r in zip(np.atleast_1d(axes), DT_OVER_TAU):
        rho = np.exp(-r)  # AR(1) over one step of size dt
        zeta = make_zeta_sequence(N_STEPS, rho)
        row = f"{r:7.2f}"
        for scheme, color in [("ab2", "C0"), ("none", "C1"), ("pc", "C2"), ("bdf2", "C3")]:
            E = run_scheme(scheme, zeta, dt, u0, force_hat, N_STEPS)
            if E is None:
                row += f" {'BLOWUP':>10}"
                ax.plot([], [], color=color, lw=1.4, label=f"{scheme} (blow-up)")
                continue
            band = E[1 : K_FORCE + 1].mean()
            ax.loglog(k[1:], E[1:] / band, color=color, lw=1.4, label=scheme)
            row += f" {E[K_TAIL].mean() / band:10.2e}"
        print(row)
        ax.set_xlabel(r"$k$")
        ax.set_title(rf"$\Delta t/\tau = {r}$", fontsize=11)
        ax.grid(alpha=0.25, which="both")
    np.atleast_1d(axes)[0].set_ylabel(r"$E(k)\,/\,\langle E\rangle_{forced}$")
    np.atleast_1d(axes)[0].legend(fontsize=9)
    fig.suptitle(
        "1-D model: equilibrium spectrum vs the noise-free remap "
        f"(CFL={CFL_FRAC}, $\\zeta_{{rms}}/c_0$={ZETA_REL})",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/model_floor_1d.png", dpi=150)
    print("saved figures/model_floor_1d.png")


if __name__ == "__main__":
    main()
