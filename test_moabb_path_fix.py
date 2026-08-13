"""Regression tests for moabb_path_fix.py's path-sanitizing patch.

These only exercise the pure pathlib logic (`_sanitize_path_preserving_anchor`,
`_sanitize_path_probe_is_buggy`, `apply`'s ImportError fallback) -- none of
which need MOABB installed, so this module has coverage even in an
environment (like this one) where `moabb` itself is not available to import.

Run: python3 test_moabb_path_fix.py
"""
from pathlib import Path

from moabb_path_fix import (
    apply,
    _sanitize_path_preserving_anchor,
    _sanitize_path_probe_is_buggy,
)


def check(name, cond):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}")
    assert cond, name


def test_illegal_chars_stripped_below_anchor():
    # POSIX absolute path: anchor is "/", which itself contains no illegal
    # character, so this exercises the same "strip below the anchor" logic
    # the drive-letter case relies on without needing a Windows Path class.
    out = _sanitize_path_preserving_anchor("/data/subj?01/file*name.mat")
    check("anchor preserved", str(out).startswith("/"))
    check("illegal chars replaced in tail", "?" not in str(out) and "*" not in str(out))
    check("exact result", str(out) == "/data/subj-01/file-name.mat")


def test_relative_path_has_empty_anchor():
    # No anchor to protect -- should sanitize the same way the original
    # (buggy) upstream function would on a relative path.
    out = _sanitize_path_preserving_anchor("subj?01/file*name.mat")
    check("relative path sanitized in full", str(out) == "subj-01/file-name.mat")


def test_probe_reports_no_bug_on_posix():
    # On POSIX, Path("C:/x").anchor == "" (no drive letters), so the probe
    # must short-circuit to False rather than false-flagging a Windows-only
    # bug on this platform.
    check("posix probe is never buggy", _sanitize_path_probe_is_buggy(lambda p: p) is False)


def test_apply_is_a_noop_without_moabb():
    # This environment does not have moabb installed; apply() must degrade
    # gracefully (return False) instead of raising ImportError.
    check("apply() returns False when moabb is unavailable", apply() is False)


if __name__ == "__main__":
    test_illegal_chars_stripped_below_anchor()
    test_relative_path_has_empty_anchor()
    test_probe_reports_no_bug_on_posix()
    test_apply_is_a_noop_without_moabb()
    print("\nAll moabb_path_fix tests passed.")
