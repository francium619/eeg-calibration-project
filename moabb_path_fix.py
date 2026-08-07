"""
Workaround for a MOABB path bug that silently redirects downloads on Windows.

MOABB 1.5.0's `moabb.datasets.download._sanitize_path` strips characters that
are illegal in Windows filenames::

    def _sanitize_path(path: Path) -> Path:
        table = {ord(c): "-" for c in ':*?"<>|'}
        return Path(str(path).translate(table))

It applies that translation to the *entire* absolute destination path, drive
letter included. On Windows the configured data root::

    C:\\Users\\<user>\\mne_data\\MNE-bnci-data\\~bci\\database\\001-2014\\A01T.mat

becomes::

    C-\\Users\\<user>\\mne_data\\MNE-bnci-data\\~bci\\database\\001-2014\\A01T.mat

which is no longer absolute. Everything is then created *relative to the
current working directory* -- so the download lands wherever the script was
launched from, `MNE_DATA` is ignored, and a re-run from a different directory
starts over rather than resuming. It fails quietly: no error, just data in the
wrong place. This repo accumulated a 13 MB partial download inside its own
working tree that way.

The colon-stripping is only meaningful for the URL-derived part of the path.
This patch preserves the path anchor (`C:\\` on Windows, `/` on POSIX) and
sanitizes only the remainder. On POSIX the anchor contains no colon, so
behavior is unchanged.

Idempotent, and a no-op if upstream fixes the bug or renames the function.
"""
from pathlib import Path

_ILLEGAL = ':*?"<>|'
_APPLIED_FLAG = "_eeg_calibration_drive_letter_patch"


def _sanitize_path_preserving_anchor(path) -> Path:
    """Sanitize illegal characters *below* the path anchor, not within it."""
    p = Path(path)
    anchor = p.anchor                      # 'C:\\', '/', or '' when relative
    rest = str(p)[len(anchor):]
    table = {ord(c): "-" for c in _ILLEGAL}
    return Path(anchor + rest.translate(table))


def apply() -> bool:
    """Patch MOABB's path sanitizer. Returns True if a patch was installed.

    Call before any MOABB download. Safe to call repeatedly.
    """
    try:
        from moabb.datasets import download as _dl
    except ImportError:
        return False

    target = getattr(_dl, "_sanitize_path", None)
    if target is None:
        return False                       # upstream renamed/removed it
    if getattr(target, _APPLIED_FLAG, False):
        return False                       # already patched

    # Only patch if the bug is actually present, so a fixed upstream is left
    # alone. Probe with a Windows-style absolute path.
    if _sanitize_path_probe_is_buggy(target):
        setattr(_sanitize_path_preserving_anchor, _APPLIED_FLAG, True)
        _dl._sanitize_path = _sanitize_path_preserving_anchor
        return True
    return False


def _sanitize_path_probe_is_buggy(fn) -> bool:
    """True if `fn` mangles a drive letter on this platform."""
    if Path("C:/x").anchor == "":
        return False                       # POSIX: no drive letters, no bug
    try:
        return "C-" in str(fn(Path("C:/probe")))
    except Exception:
        return False


def expected_destination(url: str, root) -> Path:
    """The path MOABB will use for `url` under `root`, after patching.

    Exposed so callers can assert where data is going before spending
    hundreds of megabytes finding out.
    """
    from mne.utils import _url_to_local_path
    return _sanitize_path_preserving_anchor(Path(_url_to_local_path(url, Path(root))))
