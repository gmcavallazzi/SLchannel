"""The results-folder cleanup must never destroy data slChannel did not write.

`output.clean_results_on_fresh_start` empties `output.results_folder` at the
start of a non-restart run. That is a `shutil.rmtree` over a user-supplied
path, so a mistyped config -- `results_folder: .` or a home directory -- would
otherwise delete someone's work. The cleanup only touches a folder carrying the
`.slchannel_results` sentinel that slChannel itself writes, and never one that
is the working directory, an ancestor of it, or a home/filesystem root.
"""

import os

from slchannel.solver import SLChannelFlow


def _detached(results_folder):
    """A solver object with only the attributes the cleanup path touches."""
    flow = SLChannelFlow.__new__(SLChannelFlow)
    flow.results_folder = str(results_folder)
    return flow


def test_results_guard(check, tmp_path, monkeypatch):
    # --- a folder full of someone else's files -----------------------------
    precious = tmp_path / "precious"
    (precious / "data").mkdir(parents=True)
    (precious / "thesis.tex").write_text("years of work")
    (precious / "data" / "big.npz").write_text("x")

    _detached(precious)._clean_results_folder()
    survived = sorted(p.name for p in precious.iterdir())
    check(
        "unmarked folder is not emptied",
        survived == ["data", "thesis.tex"],
        f"survivors={survived}",
    )

    # --- a folder slChannel created ----------------------------------------
    mine = tmp_path / "mine"
    mine.mkdir()
    flow = _detached(mine)
    flow._mark_results_folder()
    (mine / "fields.npz").write_text("a previous run")

    check("marker written on creation", (mine / SLChannelFlow._RESULTS_MARKER).exists())

    flow._clean_results_folder()
    left = sorted(p.name for p in mine.iterdir())
    check(
        "marked folder is emptied, marker kept",
        left == [SLChannelFlow._RESULTS_MARKER],
        f"left={left}",
    )

    # --- the working directory, even if marked -----------------------------
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "keep.txt").write_text("do not delete")
    monkeypatch.chdir(cwd)
    flow = _detached(cwd)
    flow._mark_results_folder()
    flow._clean_results_folder()
    check("working directory is refused even when marked", (cwd / "keep.txt").exists())

    # --- an ancestor of the working directory ------------------------------
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    (parent / "keep.txt").write_text("do not delete")
    monkeypatch.chdir(child)
    flow = _detached(parent)
    flow._mark_results_folder()
    flow._clean_results_folder()
    check("ancestor of the working directory is refused", (parent / "keep.txt").exists())

    # --- the home directory -------------------------------------------------
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    (fake_home / "keep.txt").write_text("do not delete")
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.chdir(tmp_path)
    flow = _detached(fake_home)
    flow._mark_results_folder()
    flow._clean_results_folder()
    check("home directory is refused even when marked", (fake_home / "keep.txt").exists())

    check("guard leaves no stray state", os.path.isdir(tmp_path))
