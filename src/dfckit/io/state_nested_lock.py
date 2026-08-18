"""Cross-process advisory locking for resumable nested workflows."""

from __future__ import annotations

import errno
import json
import os
import socket
import stat
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - exercised on POSIX
    msvcrt = None


FORMAT_NAME = "dfckit-nested-checkpoint-lock"
FORMAT_VERSION = 1


def nested_checkpoint_lock_path(checkpoint: str | Path) -> Path:
    """Return the persistent sibling lock path for one checkpoint directory."""
    target = Path(checkpoint)
    return target.parent / f".{target.name}.dfckit.lock"


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON field in nested lock: {key}")
        output[key] = value
    return output


def _invalid_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant in nested lock: {value}")


def _validate_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise TypeError("nested lock payload must be a JSON object")
    expected = {
        "acquired_at_unix",
        "format",
        "format_version",
        "hostname",
        "pid",
        "process_start_token",
        "released_at_unix",
        "state",
    }
    if set(payload) != expected:
        raise ValueError("nested lock fields do not match the schema")
    if payload["format"] != FORMAT_NAME or payload["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported nested lock format or version")
    if payload["state"] not in {"active", "released"}:
        raise ValueError("nested lock state is invalid")
    if not isinstance(payload["hostname"], str) or not payload["hostname"].strip():
        raise ValueError("nested lock hostname must be non-empty")
    pid = payload["pid"]
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        raise ValueError("nested lock pid must be a positive integer")
    token = payload["process_start_token"]
    if token is not None and (not isinstance(token, str) or not token.strip()):
        raise ValueError("nested lock process_start_token must be null or non-empty")
    acquired = payload["acquired_at_unix"]
    if (
        isinstance(acquired, bool)
        or not isinstance(acquired, (int, float))
        or not float(acquired) >= 0.0
    ):
        raise ValueError("nested lock acquired_at_unix must be non-negative")
    released = payload["released_at_unix"]
    if released is not None and (
        isinstance(released, bool)
        or not isinstance(released, (int, float))
        or not float(released) >= float(acquired)
    ):
        raise ValueError("nested lock released_at_unix is invalid")
    if (payload["state"] == "active") != (released is None):
        raise ValueError("nested lock state and release time disagree")
    try:
        json.dumps(payload, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("nested lock payload must contain finite JSON values") from error
    return payload


def _process_start_token(pid: int) -> str | None:
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        closing = stat_text.rfind(")")
        if closing < 0:
            return None
        fields_after_name = stat_text[closing + 2 :].split()
        start_ticks = fields_after_name[19]
    except (OSError, IndexError):
        return None
    return f"linux-proc:{boot_id}:{start_ticks}"


def _current_payload() -> dict[str, object]:
    pid = os.getpid()
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "state": "active",
        "pid": pid,
        "hostname": socket.gethostname(),
        "process_start_token": _process_start_token(pid),
        "acquired_at_unix": time.time(),
        "released_at_unix": None,
    }


def _open_lock(path: Path, *, create: bool) -> int:
    flags = os.O_RDWR | (os.O_CREAT if create else 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileNotFoundError:
        raise
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ValueError("nested checkpoint lock path must not be a symlink") from error
        if error.errno == errno.EISDIR:
            raise ValueError(
                "nested checkpoint lock path must be a regular file"
            ) from error
        raise
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError("nested checkpoint lock path must be a regular file")
    return descriptor


def _try_lock(descriptor: int) -> bool:
    if fcntl is not None:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True
    if msvcrt is None:  # pragma: no cover - supported platforms provide one backend
        raise RuntimeError("no supported advisory-lock backend is available")
    if os.fstat(descriptor).st_size == 0:  # pragma: no cover - Windows
        os.write(descriptor, b"\0")
        os.fsync(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    try:  # pragma: no cover - Windows
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
    except OSError as error:
        if error.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
            return False
        raise
    return True


def _unlock(descriptor: int) -> None:
    if fcntl is not None:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return
    if msvcrt is None:  # pragma: no cover
        raise RuntimeError("no supported advisory-lock backend is available")
    os.lseek(descriptor, 0, os.SEEK_SET)  # pragma: no cover - Windows
    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)  # pragma: no cover - Windows


def _read_payload(descriptor: int) -> dict[str, object]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    try:
        payload = json.loads(
            b"".join(chunks).decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read nested lock payload: {error}") from error
    return _validate_payload(payload)


def _write_payload(descriptor: int, payload: dict[str, object]) -> None:
    validated = _validate_payload(payload)
    encoded = (
        json.dumps(validated, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    offset = 0
    while offset < len(encoded):
        offset += os.write(descriptor, encoded[offset:])
    os.fsync(descriptor)


@dataclass(frozen=True)
class NestedCheckpointLockStatus:
    """Snapshot of kernel ownership and the persistent owner record."""

    path: Path
    status: str
    pid: int | None
    hostname: str | None
    process_start_token: str | None
    acquired_at_unix: float | None
    released_at_unix: float | None


class NestedCheckpointBusyError(RuntimeError):
    """Raised when another process owns the checkpoint lock."""


def _status(path: Path, state: str, payload: dict[str, object] | None) -> NestedCheckpointLockStatus:
    return NestedCheckpointLockStatus(
        path=path,
        status=state,
        pid=None if payload is None else int(payload["pid"]),
        hostname=None if payload is None else str(payload["hostname"]),
        process_start_token=(
            None if payload is None else payload["process_start_token"]
        ),
        acquired_at_unix=(
            None if payload is None else float(payload["acquired_at_unix"])
        ),
        released_at_unix=(
            None
            if payload is None or payload["released_at_unix"] is None
            else float(payload["released_at_unix"])
        ),
    )


def inspect_nested_checkpoint_lock(checkpoint: str | Path) -> NestedCheckpointLockStatus:
    """Inspect ownership without changing the lock file or checkpoint tree."""
    path = nested_checkpoint_lock_path(checkpoint)
    try:
        descriptor = _open_lock(path, create=False)
    except FileNotFoundError:
        return _status(path, "idle", None)
    acquired = False
    try:
        acquired = _try_lock(descriptor)
        if not acquired:
            try:
                payload = _read_payload(descriptor)
            except (OSError, TypeError, ValueError):
                payload = None
            return _status(path, "active", payload)
        try:
            payload = _read_payload(descriptor)
        except (OSError, TypeError, ValueError):
            return _status(path, "invalid", None)
        state = "stale" if payload["state"] == "active" else "idle"
        return _status(path, state, payload)
    finally:
        if acquired:
            _unlock(descriptor)
        os.close(descriptor)


@contextmanager
def acquire_nested_checkpoint_lock(checkpoint: str | Path) -> Iterator[Path]:
    """Exclusively own a checkpoint; kernel release makes crash recovery safe."""
    path = nested_checkpoint_lock_path(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = _open_lock(path, create=True)
    acquired = False
    payload = _current_payload()
    try:
        acquired = _try_lock(descriptor)
        if not acquired:
            try:
                owner = _read_payload(descriptor)
            except (OSError, TypeError, ValueError):
                owner = None
            detail = (
                ""
                if owner is None
                else f" by pid {owner['pid']} on {owner['hostname']}"
            )
            raise NestedCheckpointBusyError(
                f"nested checkpoint is already active{detail}: {checkpoint}"
            )
        _write_payload(descriptor, payload)
        yield path
    finally:
        if acquired:
            released = dict(payload)
            released["state"] = "released"
            released["released_at_unix"] = max(
                time.time(),
                float(payload["acquired_at_unix"]),
            )
            try:
                _write_payload(descriptor, released)
            finally:
                _unlock(descriptor)
        os.close(descriptor)


__all__ = [
    "NestedCheckpointBusyError",
    "NestedCheckpointLockStatus",
    "acquire_nested_checkpoint_lock",
    "inspect_nested_checkpoint_lock",
    "nested_checkpoint_lock_path",
]
