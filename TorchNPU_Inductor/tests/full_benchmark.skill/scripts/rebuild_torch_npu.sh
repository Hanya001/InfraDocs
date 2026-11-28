#!/usr/bin/env bash
# Rebuild torch_npu from the CURRENT pytorch checkout and install it.
# Follows 21_torch_npu_213.sh, but never checks out another commit, builds first and
# only then replaces the installed package, and keeps earlier wheels aside.
set -euo pipefail
BASE=/home/z00896543/validation/214build_20261120_1546
WS=$BASE/inductor-meta-worktree
cd "$WS"
source .claude/skills/env/env.sh
set -x
echo "HEAD $(git -C pytorch log -1 --format='%H %ci %s')"
echo "DIRTY $(git -C pytorch status --short | wc -l) files"
# sync submodules to HEAD's pins + clean codegen leftovers, as the deployment scripts do
git -C pytorch submodule update --init --recursive
git -C pytorch/third_party/op-plugin clean -fdx && git -C pytorch/third_party/op-plugin checkout -- .
git -C pytorch/third_party/dvm/dvm clean -fdx && git -C pytorch/third_party/dvm/dvm checkout -- . 2>/dev/null || true
git -C pytorch/third_party/torchair/torchair clean -fdx && git -C pytorch/third_party/torchair/torchair checkout -- . 2>/dev/null || true
git -C pytorch clean -fd torch_npu/csrc/aten
rm -rf pytorch/third_party/acl/inc/acl
mkdir -p pytorch/dist_prev_20261128 && mv pytorch/dist/*.whl pytorch/dist_prev_20261128/ 2>/dev/null || true
rm -rf pytorch/build pytorch/dist
export TORCH_NPU_ARCH=dav-3510
cd pytorch
echo "BUILD_START $(date -Is)"
bash ci/build.sh --python=3.12 --torch=2.13.0.post1
echo "BUILD_END $(date -Is)"
cd "$WS"
WHEEL=$(ls -t pytorch/dist/torch_npu-*.whl | head -1)
echo "WHEEL $WHEEL"
pip uninstall torch_npu -y
pip install "$WHEEL" --force-reinstall --no-deps
source env.sh
python -c "import torch,torch_npu; print(torch.__version__, torch_npu.__version__); \
  print('op:', torch.ops.npu._npu_moe_token_unpermute); \
  from torch_npu._inductor.triton_experimental import _activate; print('triton_experimental OK')"
"$PY" "$BASE/scripts/te_smoke.py"
cd pytorch/test/_inductor
env -u TORCHINDUCTOR_NPU_BACKEND python -m pytest test_triton_experimental_enable.py -v
echo "TORCH_NPU_REBUILD_PASS $(date -Is)"
