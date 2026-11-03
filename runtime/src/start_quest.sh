#!/bin/bash
set -euo pipefail
mkdir -p /workspace/outputs /workspace/datasets /workspace/logs51
# Driver 580 enumerates inaccessible host GPUs during NVENC initialization.
# Keep the workaround process-local to CloudXR; GPU cgroup isolation is unchanged.
env LD_PRELOAD=/workspace/encoder/libvisible_gpu_filter.so AIWORKER_NVENC_DEBUG=1 \
  AIWORKER_NVENC_GPU=GPU-cde23916-c5fa-f18e-a343-127f732cc2f6 \
  CUDA_VISIBLE_DEVICES=GPU-cde23916-c5fa-f18e-a343-127f732cc2f6 \
  /workspace/encoder/nvenc_probe > /workspace/outputs/nvenc-startup-check.log 2>&1
env LD_PRELOAD=/workspace/encoder/libvisible_gpu_filter.so AIWORKER_NVENC_DEBUG=1 \
  AIWORKER_NVENC_GPU=GPU-cde23916-c5fa-f18e-a343-127f732cc2f6 \
  CUDA_VISIBLE_DEVICES=GPU-cde23916-c5fa-f18e-a343-127f732cc2f6 \
  NV_CXR_SERVER_LOG=debug \
  /workspace/venvs/isaacteleop/bin/python -u /workspace/src/launch_cloudxr.py \
  --cloudxr-install-dir /workspace/cloudxr --accept-eula \
  > /workspace/outputs/cloudxr-launch.log 2>&1 &
cxr_pid=$!
trap 'kill -TERM "$cxr_pid" 2>/dev/null || true' EXIT
for attempt in $(seq 1 60); do
  if ! kill -0 "$cxr_pid" 2>/dev/null; then
    cat /workspace/outputs/cloudxr-launch.log
    exit 1
  fi
  if curl -ksf https://127.0.0.1:48322/ >/dev/null; then break; fi
  sleep 1
done
curl -ksf https://127.0.0.1:48322/ >/dev/null
# Require real signaling readiness before the simulator attaches to OpenXR.
# The bounded check ACKs protocol messages and releases the single client slot.
/workspace/venvs/isaacteleop/bin/python /workspace/src/check_cloudxr_signaling.py
source /workspace/cloudxr/run/cloudxr.env
export XDG_RUNTIME_DIR=/workspace/cloudxr/run
export PYTHONPATH=/workspace/pydeps51:/workspace/robotis_lab/source/robotis_lab:/workspace/robotis_lab/third_party/IsaacLab/source/isaaclab:/workspace/robotis_lab/third_party/IsaacLab/source/isaaclab_tasks
export ROBOTIS_POUR_ENABLE_CAMERAS=1
export ROBOTIS_QUEST3_DEBUG=0
export OMNICLIENT_HUB_MODE=disabled
/isaac-sim/python.sh -u /workspace/src/run_quest.py --allow-root
