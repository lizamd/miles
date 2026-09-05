"""Relax torch's triton pin when the installed triton cannot satisfy it.

The ROCm 10 / gfx1250 sglang base replaces torch's triton with its own build under a
different local version, so ``Requires-Dist: triton==<torch's build>`` in torch's metadata
names a distribution that exists on no index. pip re-resolves torch's requirements on any
install that depends on torch, so the whole image build stops there.

Rewriting the pin to a bare ``triton`` requirement is the narrowest fix: the wheel on disk is
the one the base image deliberately built for this GPU, and no version of it would ever be
downloaded anyway. A base whose pin is already satisfiable is left exactly as it is.
"""

from __future__ import annotations

import pathlib
import re
import sys
import sysconfig

PIN = re.compile(r"^Requires-Dist: triton\s*==\s*(?P<version>\S+)\s*$", re.MULTILINE)


def installed_triton_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("triton")
    except Exception:
        return None


def main() -> int:
    site = pathlib.Path(sysconfig.get_paths()["purelib"])
    metadatas = sorted(site.glob("torch-*.dist-info/METADATA"))
    if not metadatas:
        print("[torch-meta] no torch dist-info found; nothing to do")
        return 0

    have = installed_triton_version()
    for meta in metadatas:
        text = meta.read_text(encoding="utf-8")
        match = PIN.search(text)
        if not match:
            print(f"[torch-meta] {meta.parent.name}: no exact triton pin; left alone")
            continue
        want = match.group("version")
        if have == want:
            print(f"[torch-meta] {meta.parent.name}: pin triton=={want} is satisfied; left alone")
            continue
        meta.write_text(PIN.sub("Requires-Dist: triton", text), encoding="utf-8")
        print(
            f"[torch-meta] {meta.parent.name}: relaxed triton=={want} to a bare requirement "
            f"(installed: {have})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
