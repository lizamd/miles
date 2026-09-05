set -eux
export NVTE_FRAMEWORK=pytorch NVTE_USE_ROCM=1 NVTE_USE_HIPBLASLT=1
export NVTE_ROCM_ARCH=gfx1250 PYTORCH_ROCM_ARCH=gfx1250 GPU_ARCH=gfx1250
export NVTE_FUSED_ATTN=0 MAX_JOBS=64
export ROCM_PATH=${ROCM_HOME} CMAKE_PREFIX_PATH=/opt/rocm:/opt/rocm/hip:/usr/local:/usr
pip install -q ninja pybind11 onnxscript
rm -rf /w/te_build/TransformerEngine
git clone --recursive https://github.com/ROCm/TransformerEngine.git /w/te_build/TransformerEngine
cd /w/te_build/TransformerEngine
git checkout ebbd623b2403e2706bc7eff20b5487e34683f1f6
git submodule update --init --recursive
pip wheel . --no-build-isolation --no-deps -w /w/te_build/wheels -v
ls -lh /w/te_build/wheels/
