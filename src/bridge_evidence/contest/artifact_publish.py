"""Exclusive publication on POSIX and tested OSS mounts without hard links."""

import errno
import fcntl
import os
from pathlib import Path

from .run_state import file_digest, json_digest


def publish_no_replace(source: Path, target: Path, lock_dir: Path) -> dict:
    source, target, lock_dir = map(Path, (source, target, lock_dir))
    before = file_digest(source)
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_dir / (json_digest(str(target.resolve())) + ".lock")
    with lock.open("a+b") as handle:
        # All publishers use the same lock, including the POSIX fast path.
        fcntl.flock(handle, fcntl.LOCK_EX)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {target}")
        mode = "hardlink"
        try:
            os.link(source, target)
        except OSError as error:
            if error.errno not in (
                errno.EOPNOTSUPP,
                errno.ENOSYS,
                errno.EPERM,
                errno.EXDEV,
            ):
                raise
            mode = "locked-replace"
            # Native rename/replace is tested on the actual mount before formal use.
            os.replace(source, target)
        if file_digest(target) != before:
            raise RuntimeError("Published artifact bytes differ from completed source")
        stat = target.stat()
        return {
            "sha256": before,
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "mode": mode,
        }
