"""Collapse byte-identical duplicate libraries in the pip ROCm SDK onto one inode.

The SDK ships the same shared objects under several package directories -- ``_rocm_sdk_core``,
``_rocm_sdk_devel`` and ``_rocm_sdk_libraries`` -- as separate files with identical contents.
glibc's dynamic loader decides "already loaded?" by (device, inode), so two inodes holding the
same bytes are two distinct objects to it. A process that reaches one copy down one search
path and the other down another ends up with the library initialised twice. For
``libamd_comgr.so.3``, which embeds LLVM, the second initialisation aborts the process:

    CommandLine Error: Option 'spirv-expand-step' registered more than once!
    LLVM ERROR: inconsistency in registered CommandLine options

miles' Ray workers hit that while importing torch, so every Megatron-backend run dies before
it starts, while a plain `python -c "import torch"` is fine -- the two paths only meet under
the workers' import order.

Replacing the duplicates with symlinks to one canonical file makes the loader dedupe again.
Symlinks rather than hardlinks: a hardlink does not survive Docker's layering, since overlayfs
breaks the link when a later layer copies the file up, and the fix then silently disappears
from the built image. Only files whose bytes already match are touched, so nothing any library
resolves to can change.
"""

from __future__ import annotations

import collections
import filecmp
import os
import pathlib
import sys
import sysconfig

# The copy every symlink points at. _rocm_sdk_core is the SDK's own runtime package and is
# what the stock libraries' RPATHs already reach for.
CANONICAL_PACKAGE = "_rocm_sdk_core"
SKIP_SUFFIXES = (".bak",)


def main() -> int:
    site = pathlib.Path(sysconfig.get_paths()["purelib"])
    by_name: dict[str, list[pathlib.Path]] = collections.defaultdict(list)
    for path in site.glob("_rocm_sdk*/lib/*.so*"):
        if path.is_symlink() or not path.is_file():
            continue
        if path.name.endswith(SKIP_SUFFIXES):
            continue
        by_name[path.name].append(path)

    linked = reclaimed = 0
    for name, paths in sorted(by_name.items()):
        if len(paths) < 2:
            continue
        canonical = next((p for p in paths if CANONICAL_PACKAGE in p.parts), None)
        if canonical is None:
            continue
        for other in sorted(paths):
            if other == canonical or os.path.samefile(other, canonical):
                continue
            if not filecmp.cmp(other, canonical, shallow=False):
                print(f"[sdk-dedupe] {other} differs from the canonical copy; left alone")
                continue
            size = other.stat().st_size
            target = os.path.relpath(canonical, other.parent)
            tmp = other.with_name(other.name + ".dedupe-tmp")
            tmp.symlink_to(target)
            os.replace(tmp, other)
            assert os.path.samefile(other, canonical), other
            linked += 1
            reclaimed += size

    print(f"[sdk-dedupe] {linked} duplicate(s) collapsed, {reclaimed / 1e9:.2f} GB reclaimed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
