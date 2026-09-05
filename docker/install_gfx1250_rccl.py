"""Replace every librccl in the pip-installed ROCm SDK with a build that has gfx1250 code.

The ROCm 10 SDK ships librccl.so.1 as a ~4 MB stub whose ``.hip_fatbin`` is NOBITS -- no
device code for any architecture. Collectives small enough to avoid the device kernel look
correct; a 1 MiB all_reduce returns corrupted data and then faults in ncclDevKernel_Generic.

The SDK spreads the same library over more than one package directory (``_rocm_sdk_devel``
and ``_rocm_sdk_libraries``, plus per-arch ``_rocm_sdk_libraries_gfx*`` on some builds), and
the one the loader actually picks is *not* the one ``ROCM_HOME`` points at. Replacing only
``$ROCM_HOME/lib/librccl.so.1`` therefore looks like it worked and changes nothing, so this
walks every copy. Hardlinked siblings (``librccl.so``, ``.so.1``, ``.so.1.0``) resolve to one
inode and are written once.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys
import sysconfig


BACKUP_SUFFIX = ".stock-nodevicecode.bak"
RPATH_LINE = re.compile(r"Library (?:rpath|runpath): \[(?P<value>.*)\]")


def rpath_of(path: pathlib.Path) -> str | None:
    """The RPATH/RUNPATH recorded in an ELF, or None when it has neither."""
    out = subprocess.run(
        ["readelf", "-d", str(path)], capture_output=True, text=True, check=True
    ).stdout
    match = RPATH_LINE.search(out)
    return match.group("value") if match else None


def has_device_code(path: pathlib.Path) -> bool:
    """True when the ELF's .hip_fatbin occupies file bytes (PROGBITS) rather than being NOBITS."""
    out = subprocess.run(
        ["readelf", "-S", str(path)], capture_output=True, text=True, check=True
    ).stdout
    lines = out.splitlines()
    for n, line in enumerate(lines):
        if ".hip_fatbin" in line:
            return "PROGBITS" in line or (n + 1 < len(lines) and "PROGBITS" in lines[n + 1])
    return False


def main(argv: list[str]) -> int:
    src = pathlib.Path(argv[1]).resolve()
    if not src.is_file():
        print(f"[rccl] donor {src} is missing", file=sys.stderr)
        return 1
    if not has_device_code(src):
        print(f"[rccl] donor {src} has no device code (.hip_fatbin is NOBITS)", file=sys.stderr)
        return 1

    site = pathlib.Path(sysconfig.get_paths()["purelib"])
    targets = {
        p.resolve()
        for p in site.glob("_rocm_sdk*/lib/librccl.so*")
        if p.is_file() and not p.name.endswith(BACKUP_SUFFIX)
    }
    if not targets:
        print("[rccl] no librccl found under the pip ROCm SDK", file=sys.stderr)
        return 1

    for dst in sorted(targets):
        backup = dst.with_name(dst.name + BACKUP_SUFFIX)
        if not backup.exists():
            shutil.copy2(dst, backup)
        shutil.copy2(src, dst)
        dst.chmod(0o755)
        assert has_device_code(dst), dst

        # Restore the SDK's own RPATH. The stock library points at ../../_rocm_sdk_core/lib so
        # that every ROCm dependency resolves to one consistent set; the donor carries a
        # different RUNPATH from its own image, under which libamd_comgr.so.3 gets loaded from
        # _rocm_sdk_devel while another component already loaded the byte-identical copy in
        # _rocm_sdk_core. glibc dedupes by (device, inode), so two separate inodes mean two
        # LLVMs in one process, and the second one aborts with
        # "Option 'spirv-expand-step' registered more than once".
        want = rpath_of(backup)
        if want:
            subprocess.run(["patchelf", "--set-rpath", want, str(dst)], check=True)
            got = rpath_of(dst)
            assert got == want, f"{dst}: rpath is {got!r}, wanted {want!r}"
            print(f"[rccl] restored the SDK rpath on {dst.name}")
        else:
            print(f"[rccl] {dst.name}: stock library had no rpath; leaving the donor's alone")

        print(f"[rccl] installed gfx1250 RCCL at {dst} ({dst.stat().st_size} bytes)")

    # Anything still stubbed would be picked up by the loader ahead of what was just written.
    leftover = [
        p
        for p in site.glob("_rocm_sdk*/lib/librccl.so*")
        if p.is_file() and not p.name.endswith(BACKUP_SUFFIX) and not has_device_code(p)
    ]
    if leftover:
        print(f"[rccl] these copies still have no device code: {leftover}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
