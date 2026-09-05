"""Stop torch trusting a zero device count from amdsmi when HIP disagrees.

``torch.cuda.device_count()`` prefers ``_raw_device_count_amdsmi()`` and only falls back to
``hipGetDeviceCount`` when that returns a *negative* value. On a gfx1250 node whose librccl
carries real device code, ``amdsmi_get_processor_handles()`` comes back empty -- RCCL brings
libamd_smi up itself, and the later init from torch's ``amdsmi`` module then enumerates
nothing. Zero is not negative, so torch reports zero GPUs while HIP happily reports four, and
everything that asks torch how many devices exist (Megatron, Transformer Engine's
``get_device_compute_capability``, miles' own hardware detection) fails on a machine that
otherwise runs fine. SGLang's own gfx1250 stage hit the same class of problem and dealt with
it by not installing amd_smi at all.

Zero handles alongside a working HIP is a discovery failure, not an answer, so this rewrites
that one ``return`` to report it the way every other failure in the function is reported.
Nothing else about amdsmi changes: the module stays installed and usable.
"""

from __future__ import annotations

import pathlib
import sys

NEEDLE = """    socket_handles = amdsmi.amdsmi_get_processor_handles()
    return len(socket_handles)"""

REPLACEMENT = '''    socket_handles = amdsmi.amdsmi_get_processor_handles()
    # Patched for gfx1250: an empty handle list means amdsmi discovery failed, not that the
    # machine has no GPUs. Report it as a failure (-1) so device_count() falls back to
    # hipGetDeviceCount instead of caching a zero.
    if not socket_handles:
        return -1
    return len(socket_handles)'''


def main() -> int:
    import torch.cuda

    target = pathlib.Path(torch.cuda.__file__)
    text = target.read_text(encoding="utf-8")
    if REPLACEMENT in text:
        print(f"[amdsmi] {target} already patched")
        return 0
    if NEEDLE not in text:
        print(
            f"[amdsmi] _raw_device_count_amdsmi in {target} does not look the way this patch "
            "expects; refusing to guess",
            file=sys.stderr,
        )
        return 1
    target.write_text(text.replace(NEEDLE, REPLACEMENT, 1), encoding="utf-8")
    for cache in target.parent.glob("__pycache__/__init__.*.pyc"):
        cache.unlink()
    print(f"[amdsmi] patched {target}: empty handle list now falls back to hipGetDeviceCount")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
