"""Trusted namespace launcher; no candidate code runs before chroot and seccomp."""

import ctypes
import os
import sys

WORKER_ENV = {
    "LD_LIBRARY_PATH": "/lib:/usr/local/lib:/usr/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu"
}


def main():
    libc = ctypes.CDLL(None, use_errno=True)
    parent = os.getppid()
    if libc.prctl(1, 9, 0, 0, 0) or parent == 1 or os.getppid() != parent:
        raise RuntimeError("sandbox parent-death protection failed")
    if sys.argv[1] == "--outer":
        launcher = os.path.realpath(__file__)
        os.execve(
            "/usr/bin/unshare",
            [
                "unshare",
                "--user",
                "--map-root-user",
                "--mount",
                "--net",
                "--pid",
                "--fork",
                "--ipc",
                "--uts",
                sys.executable,
                "-I",
                "-S",
                "-B",
                launcher,
                sys.argv[2],
            ],
            {"PATH": "/usr/bin:/bin"},
        )
    root = os.path.realpath(sys.argv[1])
    libc.mount.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_ulong,
        ctypes.c_void_p,
    ]

    def mount(source, target, flags):
        if libc.mount(
            source.encode() if source else None, target.encode(), None, flags, None
        ):
            raise OSError(ctypes.get_errno(), "sandbox mount failed")

    mount(None, "/", (1 << 14) | (1 << 18))  # recursive private propagation
    mount(root, root, 4096)
    mount(None, root, 4096 | 32 | 1 | 2 | 4)  # bind-remount readonly,nosuid,nodev
    os.chroot(root)
    os.chdir("/")
    for fd in range(3, min(65536, os.sysconf("SC_OPEN_MAX"))):
        try:
            os.close(fd)
        except OSError:
            pass

    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

    class Data(ctypes.Structure):
        _fields_ = [
            ("effective", ctypes.c_uint32),
            ("permitted", ctypes.c_uint32),
            ("inheritable", ctypes.c_uint32),
        ]

    if libc.capset(ctypes.byref(Header(0x20080522, 0)), ctypes.byref((Data * 2)())):
        raise OSError(ctypes.get_errno(), "capability drop failed")
    if libc.prctl(38, 1, 0, 0, 0):
        raise RuntimeError("no_new_privs failed")
    os.execve("/bin/python", ["python", "-I", "-S", "-B", "/worker.py"], WORKER_ENV)


if __name__ == "__main__":
    main()
