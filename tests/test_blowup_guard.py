"""The blow-up guard: three consecutive over-threshold diagnostics stop the
run; isolated spikes and recoveries do not.

Pure-logic test of SLChannelFlow._blowup_check -- no solver construction,
no GPU. The physical rationale: a blown SL run saturates at finite
amplitude (foot clamping plus the exact flux constraint), so the NaN check
never fires; u_tau roughly doubles instead, and that is the observable.
"""

from types import SimpleNamespace

from slchannel.solver import SLChannelFlow


def make_state(nominal=0.064, factor=2.0):
    return SimpleNamespace(
        _u_tau_nominal=nominal, blowup_u_tau_factor=factor, _blowup_strikes=0
    )


def check(state, u_tau):
    return SLChannelFlow._blowup_check(state, u_tau)


def test_healthy_run_never_triggers():
    s = make_state()
    for u_tau in [0.061, 0.064, 0.067, 0.070, 0.060] * 10:
        assert not check(s, u_tau)


def test_sustained_blowup_triggers_on_third_strike():
    s = make_state()
    assert not check(s, 0.14)
    assert not check(s, 0.15)
    assert check(s, 0.15)


def test_single_spike_resets():
    s = make_state()
    assert not check(s, 0.20)
    assert not check(s, 0.20)
    assert not check(s, 0.063)  # recovery resets the strikes
    assert not check(s, 0.20)
    assert not check(s, 0.20)
    assert check(s, 0.20)


def test_disabled_before_nominal_is_known():
    s = make_state(nominal=None)
    for _ in range(5):
        assert not check(s, 1.0)
