set -eux
export PATH=/opt/rocm/bin:$PATH
export ROCM_PATH=${ROCM_HOME} HIP_PATH=${ROCM_HOME}
export PYTORCH_ROCM_ARCH=gfx1250 GPU_ARCH=gfx1250 AMDGPU_TARGET=gfx1250
export MAX_JOBS=64
python3 /w/miles/docker/relax_torch_triton_pin.py || true
cd /w/apex_build
rm -rf apex && git clone --depth 1 https://github.com/ROCm/apex.git
cd apex
git log --oneline -1
# The ROCm apex fork gates its extensions on environment variables, not on the upstream
# --cpp_ext / --cuda_ext argv flags (which it strips and ignores). Passing the flags -- via
# pip --build-option or setup.py directly -- silently yields a pure-python wheel, so the
# assertion below is what actually proves the build did anything.
export APEX_BUILD_CPP_OPS=1 APEX_BUILD_CUDA_OPS=1
rm -rf /w/apex_build/wheels && mkdir -p /w/apex_build/wheels
python setup.py bdist_wheel 2>&1 | tail -40
cp dist/apex-*.whl /w/apex_build/wheels/
python3 - <<'PY'
import glob, zipfile
w = sorted(glob.glob("/w/apex_build/wheels/apex-*.whl"))[-1]
so = [n for n in zipfile.ZipFile(w).namelist() if n.endswith(".so")]
print(f"[apex] {w}: {len(so)} extension module(s)")
for n in so: print("   ", n)
assert any("fused_weight_gradient_mlp_cuda" in n for n in so), \
    "fused_weight_gradient_mlp_cuda was not compiled; the wheel is useless for Megatron"
PY
ls -lh /w/apex_build/wheels/
