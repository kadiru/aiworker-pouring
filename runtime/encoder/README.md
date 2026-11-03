# AIWorker encoder startup repair

October 28, 2026 PDT: the Quest reached the HTTPS/WSS proxy at 192.168.68.60, but video did not start. CloudXR's log contained repeated `NV_ENC_ERR_UNSUPPORTED_DEVICE` errors. This matched the [reported multi-GPU enumeration issue](https://github.com/NVIDIA/nvidia-container-toolkit/issues/1249) with NVIDIA driver 580. The container already had the correct driver libraries and `NVIDIA_DRIVER_CAPABILITIES=all`.

`nvenc_probe.c` verifies that CUDA sees exactly one GPU with the expected RTX 4070 UUID, opens a CUDA context, opens/closes an NVENC session and queries the codec count. Before the fix it returned error 2; with the fix it returned success and three codecs. This is an encoder initialization check, not an end-to-end streaming test.

`visible_gpu_filter.c` is a process-local library loaded only for that startup check and CloudXR. It forwards the actual driver call, then narrows a successful `GPU_GET_ATTACHED_IDS` reply to the one device node assigned to the container. It checks the exact ioctl ABI, descriptor type, response size/status and NVIDIA's documented `GET_ID_INFO_V2` mapping. It does not guess PCI identifiers or change the kernel, driver, permissions, cgroups, other processes or GPU allocation. If mapping cannot be established, it leaves the response unchanged, so the encoder may fail. The 4090 device node remains absent.

The library activates only with `AIWORKER_NVENC_GPU=GPU-cde23916-c5fa-f18e-a343-127f732cc2f6`. Do not use `NVIDIA_VISIBLE_DEVICES` as its activation flag: this host's container startup changes that value to `void` even though Docker exec reports the configured UUID. Explicit CUDA visibility and the probe's UUID assertion add checks; Docker's unchanged GPU-device allocation remains the actual isolation boundary.

Build on Ubuntu inside the project directory (CPU compilation only):

```sh
cd /home/kadir/aiworker-pouring-4070/encoder
gcc -Wall -Wextra -Werror -O2 -o nvenc_probe nvenc_probe.c -ldl
gcc -Wall -Wextra -Werror -O2 -shared -fPIC -o libvisible_gpu_filter.so visible_gpu_filter.c -ldl -pthread
```

`src/start_quest.sh` runs the probe first and refuses to start if encoding fails. It passes `LD_PRELOAD` only to the probe and CloudXR process. It does not export it to Isaac Sim, other containers, the host or the humanoid environment. Remove those process-local settings and the preflight to revert; driver files need no restoration. Restart through the identity-checking, GPU-guarded launcher only.

Files downloaded for API definitions and source inspection:

- `nvEncodeAPI.h`: NVIDIA's MIT-licensed API header from [FFmpeg nv-codec-headers n12.0.16.1](https://github.com/FFmpeg/nv-codec-headers/blob/n12.0.16.1/include/ffnvcodec/nvEncodeAPI.h), used by the probe; notice retained in the file.
- `ctrl0000gpu.reference.h`: [NVIDIA open kernel module definitions 580.95.05](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.95.05/src/common/sdk/nvidia/inc/ctrl/ctrl0000/ctrl0000gpu.h), used to inspect the response structures, not compiled as a kernel module.
- `nvenc_fix.reference.c`: [flexgrip's diagnostic reference](https://github.com/flexgrip/nvidia-gpu-enumeration), read but not compiled or executed. Its GPL-3.0 license is retained in `REFERENCE-LICENSE`. The deployed implementation uses the documented V2 mapping and has no PCI-bit guessing fallback.

Evidence: `../../evidence/nvenc-probe-comparison.txt`, `../../evidence/quest-encoder-repair.json`. The fresh CloudXR log showed successful SPS/PPS preparation and no NVENC errors after the repair. Headset retry is still required to establish whether streaming and tracking now work.
