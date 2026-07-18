# Stages the X11/OpenGL runtime libraries open3d links against into a user directory, for display-less nodes without root.

from __future__ import annotations

import argparse
import platform
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

_MIRRORS = {
    "arm64": "http://ports.ubuntu.com/ubuntu-ports",
    "amd64": "http://archive.ubuntu.com/ubuntu",
}
_PACKAGES = [
    ("libx/libxau", "libxau6"),
    ("libx/libxdmcp", "libxdmcp6"),
    ("libx/libxcb", "libxcb1"),
    ("libx/libx11", "libx11-6"),
    ("libx/libxext", "libxext6"),
    ("libg/libglvnd", "libglvnd0"),
    ("libg/libglvnd", "libglx0"),
    ("libg/libglvnd", "libgl1"),
    ("libg/libglvnd", "libegl1"),
    ("libg/libglvnd", "libopengl0"),
]


def _deb_arch() -> str:
    machine = platform.machine()
    if machine in ("aarch64", "arm64"):
        return "arm64"
    if machine in ("x86_64", "amd64"):
        return "amd64"
    raise SystemExit(f"unsupported architecture {machine!r}")


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as resp:
        data: bytes = resp.read()
    return data


def _deb_url(pool_dir: str, package: str, arch: str, mirror: str) -> str:
    listing = _fetch(f"{mirror}/pool/main/{pool_dir}/").decode("utf-8", "replace")
    matches = re.findall(rf'href="({re.escape(package)}_[^"]+_{arch}\.deb)"', listing)
    if not matches:
        raise SystemExit(f"no {arch} .deb found for {package} under pool/main/{pool_dir}")
    # First listing entry is the oldest available build — the safest glibc floor for
    # whatever base image the container uses.
    return f"{mirror}/pool/main/{pool_dir}/{matches[0]}"


def stage(dest: Path) -> Path:
    arch = _deb_arch()
    mirror = _MIRRORS[arch]
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for pool_dir, package in _PACKAGES:
            url = _deb_url(pool_dir, package, arch, mirror)
            deb = Path(tmp) / url.rsplit("/", 1)[1]
            deb.write_bytes(_fetch(url))
            subprocess.run(["dpkg-deb", "-x", str(deb), str(dest)], check=True)
            print(f"staged {package} <- {url.rsplit('/', 1)[1]}")
    libdir = dest / "usr" / "lib" / f"{platform.machine()}-linux-gnu"
    print(f"library dir: {libdir}")
    return libdir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage headless X11/OpenGL libraries without root (for open3d import)."
    )
    parser.add_argument("--dest", default=str(Path.home() / ".local" / "headless-gl-libs"))
    args = parser.parse_args()
    libdir = stage(Path(args.dest))
    return 0 if libdir.is_dir() else 1


if __name__ == "__main__":
    raise SystemExit(main())
