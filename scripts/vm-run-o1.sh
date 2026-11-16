#!/usr/bin/env bash
# O1 on a rented Blackwell VM (B300 / B200). Run step by step:
#   bash vm-run-o1.sh check    # GPU, driver, disk, python; ~1 min
#   bash vm-run-o1.sh setup    # venv + torch; ~5 min
#   bash vm-run-o1.sh build    # clone + build the pinned Triton; ~20-40 min on 24 CPUs
#   bash vm-run-o1.sh ncu      # optional: can we read GPU counters on this VM? ~1 min
#   bash vm-run-o1.sh quick    # 8192^3, one-missing-scale modes + workarounds, artifacts; ~10-15 min
#   bash vm-run-o1.sh full     # all modes and shapes, artifacts; ~30-40 min (optional)
#   bash vm-run-o1.sh profile  # Nsight Compute: fallback vs workaround, same config + best configs; ~10-20 min
#   bash vm-run-o1.sh pack     # tar results to copy back
# Copy to the VM first (from the local machine):
#   scp bench_o1_fresh.py ncu_o1_fresh.py vm-run-o1.sh <vm>:~/o1/
# Nothing here pushes, posts or uploads anything; results stay on the VM until you scp them back.
set -euo pipefail

TRITON_COMMIT=ecd97a63d120a6a5862dd0002a5544e8bb392a3b   # pinned 2026-11-13; refusal checked offline on it
O1=${O1:-$HOME/o1}
VENV=${VENV:-$HOME/venv-o1}
TRITON_DIR=${TRITON_DIR:-$HOME/triton-o1}
# torch: override if the image already has a working one, e.g. TORCH_SPEC=skip
TORCH_SPEC=${TORCH_SPEC:-"torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128"}
RES=$O1/results-$(hostname)-$(date +%Y%m%d)
mkdir -p "$O1" "$RES"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$RES/steps.log"; }
act() { # shellcheck disable=SC1091
  source "$VENV/bin/activate"; }

case "${1:-}" in
check)
  # Only the NVIDIA DRIVER is required from the image. The CUDA runtime comes with the torch wheel, ptxas/cuobjdump
  # come with the Triton build, nvcc is not used, ncu is optional.
  if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
    log "STOP: no working NVIDIA driver (nvidia-smi). Pick an image with the driver preinstalled; installing it here is slow and needs a reboot."
    exit 1
  fi
  log "nvidia-smi:"; nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv | tee -a "$RES/steps.log"
  command -v g++ >/dev/null || command -v clang++ >/dev/null || { log "STOP: no C++ compiler (apt install build-essential)"; exit 1; }
  python3 -c "import sysconfig, os; assert os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h'))" 2>/dev/null \
    || { log "STOP: Python headers missing (apt install python3-dev)"; exit 1; }
  command -v ncu >/dev/null && log "ncu present: $(ncu --version | tail -1)" || log "ncu not installed (optional; SASS is collected anyway)"
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  case "$cap" in 10.0|10.3) log "OK: Blackwell datacenter GPU (compute_cap $cap)";;
    *) log "STOP: compute_cap $cap is not sm_100/sm_103 — O1 needs B200/B300"; exit 1;; esac
  log "disk free in \$HOME: $(df -h "$HOME" | tail -1 | awk '{print $4}') (want >= 100G)"
  log "cpus: $(nproc); python: $(python3 --version 2>&1)"
  command -v git >/dev/null || { log "STOP: git missing"; exit 1; }
  python3 -c "import venv" 2>/dev/null || { log "STOP: python3-venv missing (apt install python3-venv)"; exit 1; }
  ;;
setup)
  python3 -m venv "$VENV"; act
  pip install -q -U pip wheel
  if [ "$TORCH_SPEC" != "skip" ]; then
    # shellcheck disable=SC2086
    pip install -q $TORCH_SPEC
  fi
  pip install -q "cmake>=3.20,<4.0" "ninja>=1.11.1" "setuptools>=40.8.0" pybind11 numpy
  python - <<'EOF' | tee -a "$RES/steps.log"
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(), "cap", torch.cuda.get_device_capability())
x = torch.randn(256, 256, device="cuda"); print("matmul ok", float((x @ x).sum()) == float((x @ x).sum()))
EOF
  # pip may have pulled a 'triton' wheel with torch; it must NOT shadow our build
  pip uninstall -y -q triton 2>/dev/null || true
  log "setup done"
  ;;
build)
  act
  if [ ! -d "$TRITON_DIR/.git" ]; then git clone -q https://github.com/triton-lang/triton.git "$TRITON_DIR"; fi
  git -C "$TRITON_DIR" fetch -q origin
  git -C "$TRITON_DIR" checkout -q "$TRITON_COMMIT"
  log "triton at $(git -C "$TRITON_DIR" log -1 --format='%H %ci %s')"
  cd "$TRITON_DIR"
  # Last time main failed with -Werror on deprecated MLIR builders; keep that warning non-fatal.
  export CXXFLAGS="${CXXFLAGS:-} -Wno-error=deprecated-declarations"
  export MAX_JOBS=${MAX_JOBS:-$(nproc)}
  pip uninstall -y -q triton 2>/dev/null || true
  # Install the build requirements declared by THIS commit (--no-build-isolation skips pip's own resolution).
  reqs=$(python -c "import tomllib;print(' '.join(tomllib.load(open('pyproject.toml','rb'))['build-system']['requires']))")
  log "build requirements from pyproject.toml: $reqs"
  # shellcheck disable=SC2086
  pip install -q $reqs
  log "pip install -e . (MAX_JOBS=$MAX_JOBS)"
  pip install --no-build-isolation -e . > "$RES/triton-build.log" 2>&1 || { log "BUILD FAILED, see $RES/triton-build.log"; tail -30 "$RES/triton-build.log"; exit 1; }
  cd "$O1"
  python - <<EOF | tee -a "$RES/steps.log"
import triton, os, subprocess
p = os.path.abspath(triton.__file__)
print("triton imported from", p)
assert p.startswith(os.path.abspath("$TRITON_DIR")), "imported triton is NOT the pinned checkout"
print("commit", subprocess.run(["git","-C","$TRITON_DIR","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip())
EOF
  log "build done"
  ;;
ncu)
  act
  if ! command -v ncu >/dev/null; then log "ncu not on PATH (CUDA toolkit not installed?) — skip"; exit 0; fi
  ncu --version | tail -1 | tee -a "$RES/steps.log"
  set +e
  ncu --launch-count 1 --metrics sm__cycles_elapsed.avg python -c "import torch; x=torch.randn(512,512,device='cuda'); (x@x).sum().item()" > "$RES/ncu-check.log" 2>&1
  rc=$?; set -e
  if grep -q "ERR_NVGPUCTRPERM" "$RES/ncu-check.log"; then
    log "ncu: counters BLOCKED (ERR_NVGPUCTRPERM). On a VM with root: sudo sh -c 'echo options nvidia NVreg_RestrictProfilingToAdminUsers=0 > /etc/modprobe.d/ncu.conf' then reload the driver/reboot; or run ncu with sudo."
  elif [ $rc -eq 0 ] && grep -q "sm__cycles_elapsed" "$RES/ncu-check.log"; then
    log "ncu: counters AVAILABLE"
  else
    log "ncu: unclear (rc=$rc), see $RES/ncu-check.log"
  fi
  ;;
quick|full)
  act; cd "$O1"
  extra=$([ "$1" = quick ] && echo --quick || true)
  log "run $1"
  set +e
  python bench_o1_fresh.py $extra --expect-commit "$TRITON_COMMIT" --artifacts "$RES/art-$1" --out "$RES/o1_$1.txt"
  rc=$?; set -e
  log "bench $1 exit=$rc (0 ok, 1 correctness/commit failure, 2 unexpected mma kind)"
  grep -E "BEST|RANDOM|SASS|cuBLAS|FAIL|UNEXPECTED|EXIT" "$RES/o1_$1.txt" | tee -a "$RES/steps.log" | tail -40
  ;;
profile)
  # Nsight Compute on the fallback vs the workaround, at the SAME config (fair) and at each one's usual best config.
  act; cd "$O1"
  command -v ncu >/dev/null || { log "ncu not on PATH"; exit 1; }
  P="$RES/ncu"; mkdir -p "$P"
  SECTIONS="--section SpeedOfLight --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --section WarpStateStats --section SchedulerStats --section LaunchStats --section Occupancy"
  for pair in "lhsvar 128x128x64:4:4" "lhsvar+const 128x128x64:4:4" "lhsvar 128x256x128:8:3" "lhsvar+const 128x256x128:8:3" \
              "rhsvar 128x128x64:4:4" "rhsvar+const 128x128x64:4:4"; do
    set -- $pair; m=$1; c=$2; tag="${m//+/_plus_}_${c//:/_}"
    log "ncu $m $c"
    # shellcheck disable=SC2086
    ncu --kernel-name mm_kernel --launch-skip 3 --launch-count 1 $SECTIONS -f -o "$P/$tag" \
        python ncu_o1_fresh.py "$m" "$c" > "$P/$tag.log" 2>&1 || log "  ncu FAILED for $m $c (see $P/$tag.log)"
    grep -E "exact|ERR_|==ERROR==" "$P/$tag.log" | tee -a "$RES/steps.log"
    [ -f "$P/$tag.ncu-rep" ] && ncu --import "$P/$tag.ncu-rep" --page details > "$P/$tag.details.txt" 2>&1
  done
  log "profile done: $P (*.ncu-rep open in Nsight Compute GUI; *.details.txt for text)"
  ;;
pack)
  tar czf "$O1/o1-results.tgz" -C "$O1" "$(basename "$RES")"
  log "packed: $O1/o1-results.tgz  ->  scp <vm>:$O1/o1-results.tgz ."
  log "then STOP the VM in the provider console"
  ;;
*)
  sed -n '2,13p' "$0"; exit 1;;
esac
