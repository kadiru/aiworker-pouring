# Repository and external artifacts

Repository: [kadiru/aiworker-pouring](https://github.com/kadiru/aiworker-pouring), private, default branch `main`.

The initial commit captures the existing project as of November 3, 2026. Earlier work was saved as files, native recordings, checkpoints and rollback copies; it was not recorded as Git commits. Historical results are retained without inventing earlier commit history.

## Versioned here

- Project source, launchers, GPU-priority guard, viewers and tests.
- Configuration and operational documentation, including the actual limitations and failed tests.
- Small JSON/CSV reports, hashes, manifests and model-selection provenance from the learning studies.
- The minimal MIT-licensed NVIDIA encoder API header required by the probe, with its notice retained. Downloaded vendor implementations/checkouts remain external.

The working repository lives at `/Users/kadir/linkedin/projects/aiworker-pouring`. Ubuntu's `/home/kadir/aiworker-pouring-4070` remains a deployed runtime copy; creating this repository does not restart, retrain or modify the running simulator or other jobs. Bring tested source changes back to this working repository before committing them.

## External artifacts

`.gitignore` excludes recordings, images/videos, model-weight files, fitted coefficient/profile JSON, USD assets, environments, caches, vendor checkouts, downloaded SDKs, recovered chat history, rollback copies, credentials and connection keys. Existing local and Ubuntu copies are retained.

| Artifact | Existing location / identity |
|---|---|
| Active finish-study V23 candidate | Ubuntu `learning/A-finish-20261106-v1/best.pt`; local `evidence/finish-A/best.pt`; SHA `bc9dc0e2d0afd2d090ff36f283be2276a9090f5fc48d3bf8290291dfcf2ee81e`. Round-2 development only; reserved evaluation not yet frozen. |
| Finish-study r0 archive and records | Native `learning/A-finish-20261106-v1/candidates/v23-r0/` and local `evidence/finish-A/candidates/v23-r0/` preserve original checkpoint `f26ff47868da35f47c9b8a6011132b25211a0cfd9c85522a0caa1468b65647ab`. Native recordings stay Ubuntu; local `data/material-A-finish/` backs up scalar metadata only. Catalogued recorder abort remains incomplete. |
| Finish-study r1 archive | Native `learning/A-finish-20261106-v1/candidates/v23-r1/` retains checkpoint `f9ffbce349aa5b3a0720205f28c5fc51499a5a0b72762bc5470aa41b2930bb27`, sources, audit, reviewed frames manifest, motion evidence and braking probe. Local `evidence/finish-A/candidates/v23-r1/` retains weights/model; versioned r1 reports preserve its 7/8 result and overflow. |
| Frozen wider/swapped V22 candidate | Ubuntu `learning/A-wide-20261105-v1/best.pt` and `frozen-candidate/`; local `evidence/wide-A/best.pt`, SHA256 `4c8ab7bab4b5121004e47ac1458f6df8ead0bd481c381efa807c7be8195d3430`; core source commit `07cd41c321c7a00229618bc0db912a38be388001` |
| Wider/swapped model provenance | Local `data/material-A-wide-models/fit-r5/`; earlier native `fit-attempts/r0`–`r4` retain failed/superseded models and sources. Fitted coefficient JSON and weights are external. |
| Wider/swapped study recordings | Native `outputs/fountain/`; local `data/material-A-wide/`: 99 recordings and 359 sidecars, comprising 37 references, 32 development attempts and 30 fresh tests. Reviewed reference and r0/r1/r2 development HDF5 backups may be lossless `.hdf5.zst` archives; native originals stay unchanged. |
| Wider/swapped study reports and videos | Versioned `evidence/wide-A/` JSON/CSV, selection/model/source hashes, resource and review evidence. Local `thirty-wide-tests-part-1.mp4`, `-2.mp4`, `-3.mp4` include all thirty attempts; media remain external to Git. See [the study](WIDE_TABLE_STUDY.md). |
| Frozen expanded-position candidate | Ubuntu `learning/A-workspace-20261104-v1/best.pt`, SHA256 `a909abb318ecc980ad8450484ca98ba07990f9501cd641744b5ab6544732f51b` |
| Expanded candidate archive | Ubuntu `learning/A-workspace-20261104-v1/candidates/v18-r0/`; model, fit reports and exact 29-file runtime snapshot; repaired `candidates/v18-r1/` has the new model and 30 sources |
| Expanded study recordings | `data/material-A-workspace/`; 64 HDF5 and 234 sidecars checksum-verified: 22 references, four r0 development attempts, eight r1 development attempts and thirty fresh tests |
| Expanded study evidence | `evidence/workspace-A/`; verified 23/30 report and CSV, particle conservation, split/fit manifests, resource snapshots and camera review hashes; local outcome PDF and the two MP4s are external to Git |
| Completed narrow-workspace reliability model | `evidence/reliable-A/best.pt`; Ubuntu `learning/A-reliable-20261103-v1/best.pt` |
| Completed reliability model SHA256 | `40b5506bc4bceb51f1bf8ab8a5df93525933da0b03dea4c7e3fc66b63ba62101` |
| Reliability study records | `data/material-A-reliable/`; 44 complete HDF5 files and 176 sidecars, no partials, native-checksum-verified |
| Reliability provenance | `evidence/reliable-A/final-selection.json`, `model-summary.json`, `data-manifest.json`, final 30/30 report, both development rounds, integrity hashes and final review |
| Archived development candidates | Ubuntu `learning/A-reliable-20261103-v1/candidates/v16-r0/` and `candidates/v17-r1/`; models, metadata and exact 22-file source snapshots |
| Previous corrective evaluation model | `evidence/settled-A/best.pt`; Ubuntu `learning/A-settled-20261103-v1/best.pt` |
| Previous corrective model SHA256 | `18f0064ea9f1503488ba606a863a8047bf1f3cd8ff05dfb95a81b53e24121527` |
| Previous corrective study records | `data/material-A-settled/`; 48 complete HDF5 files, 192 sidecars, three partial recordings and three partial sidecars, native-checksum-verified |
| Previous corrective provenance | `evidence/settled-A/final-selection.json`, `model-summary.json`, `data-manifest.json`, final 27/30 report, all failed attempts and integrity hashes |
| Prior frozen evaluation model | `evidence/robust-A/best.pt`; Ubuntu `learning/A-robust-20261103-v1/best.pt` |
| Prior six-case repaired model | `evidence/guard-B/best.pt`; Ubuntu `learning/A-guard-20261103-v2/best.pt` |
| Prior evaluation model SHA256 | `b80ff58388e8f2516975f1925b20aa98c5720ab395bba3fe04dfb508fac64b1b` |
| Thirty-case native recordings/labels | `data/material-A-robust/`; 30 complete HDF5 and 120 JSON sidecars, native-checksum-verified |
| Thirty-case integrity/index | Versioned `evidence/robust-A/recording-integrity.json`; local `data/material-A-robust/backup-index.json` |
| Prior six-case native recordings/labels | `data/material-A-guard-vote/`; nine complete HDF5 and36JSON sidecars, native-checksum-verified |
| Prior six-case integrity/index | Versioned `evidence/guard-B/recording-integrity.json`; local `data/material-A-guard-vote/backup-index.json` |
| First guarded round | `evidence/guard-A/best.pt`, SHA256`f4ddb8b9f288ad2e5e8a9e6ac495f57a94c7413a0cdb4bbb040a89769bd36098`;19HDF5+76sidecars+one interrupted partial under `data/material-A-guard/` |
| Archived visual-stop study | `evidence/stop-A/best.pt`, SHA256`b47569e3331b39818e3bd04975c7487d9c23ece1e8a51be42133243a1e938b35`;12HDF5+48sidecars under `data/material-A-stop/` |
| Robot USD | `runtime/assets/FFW_BG2_without_mimic.usd`, SHA256 `82ac39486428e51d8be58996ea817fd4b25a40c262936e2d2cc8823673e1387d` |
| Prior reference datasets/checkpoints | Existing `data/material-A-*` and `evidence/learning-A`, `evidence/recovery-A`; metadata is versioned, binary artifacts are external |
| Simulator/runtime dependencies | Isaac Sim 5.1 and the recovered Isaac Lab/ROBOTIS runtime described in `runtime/README.md` and `docs/RECOVERED_SETUP.md` |

A fresh clone contains source and reports, not a self-contained simulator installation. Restore the external assets, datasets, selected/prerequisite checkpoints and runtime dependencies before reproducing native experiments. Some historical documentation links point to these intentionally untracked artifacts and will resolve only in a workspace with the backups restored. GitHub is not the backup destination for those artifacts.

## Routine workflow

Inspect `git status` and `git diff`, run checks appropriate to the change, make a logical commit, then push `main` to `origin`. Review staged files before pushing. Do not force-add ignored credentials, recordings, model weights, vendor trees or caches. Use the configured repository-local identity; do not change global Git identity or another project's repository settings.

CPU-only baseline checks:

```sh
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s runtime -p test_gpu_priority_guard.py -v
```

These checks do not start Isaac Sim, train a model or use the GPU. Native simulation results and their limits are documented separately in `docs/VISUAL_STOP_STUDY.md`.

Completed narrow-workspace reliability verification (use source revision `4f8c141f5db96e9b706f2e6f705baf6e9295a1c7`, or its archived source files, for source-bound checks; the expanded controller intentionally differs): `python3 tools/summarize_reliable_benchmark.py`; analysis: `python3 tools/analyze_robust_outcomes.py --study reliable`. All thirty scenes ran once after frozen source commit `76c726ea0dfb17392d879959ca9ad844e7a2334e`, with **30/30 strict passes**, fifteen full and fifteen empty. Model, 22 runtime sources, asset, resource caps and all 44 development/test recordings plus 176 sidecars are verified. [Completed study](RELIABILITY_STUDY.md). Actual full/empty videos and all five review sheets live under `evidence/reliable-A/` outside Git.

Prior six-case repaired-study verification: `python3 tools/summarize_guard_benchmark.py --study vote`. For the archived first guarded round, use source revision `9a4b8b1` so its source-bound checks match the original controller. The `guard-B` folder denotes repair round 2; vessel material remains A.

Thirty-case frozen evaluation verification: `python3 tools/summarize_robust_benchmark.py`; failure analysis: `python3 tools/analyze_robust_outcomes.py`. The unchanged checkpoint was evaluated once on 30 new reserved cases; 25 passed, five failures are retained. See [the full report](ROBUSTNESS_A_30.md). Camera sheets and full/empty/underfill videos live under `evidence/robust-A/` outside Git.

Previous corrective 30-case verification: `python3 tools/summarize_settled_benchmark.py`; analysis: `python3 tools/analyze_robust_outcomes.py --study settled`. All 30 ran once after the frozen source commit `27ba46e`: 27 passes, two localization aborts and one one-particle spill. Model, 18 runtime sources, asset and complete/partial backups are verified. [Final corrective study](SETTLED_LEVEL_STUDY.md). Use archived source revision `07a06ab` for its source-bound verifier and `b2c7600` for the previous robustness study; current controller files intentionally differ. Raw media/models are backed up locally and natively, not uploaded to GitHub.

Prepared late-drip follow-up: `tools/audit_finish_late_drip.py` and `tools/calibrate_finish_drip.py` isolate geometry screening and an idle-only, archived checkpoint/source update. `evidence/finish-A/late-drip-clearance-r3.json` covers 38 authored fixtures; `late-drip-cpu-tests-r3.json` records 24 CPU checks. The r2 case-03 scalar diagnostic, native recording hash, lip geometry and eight-frame review preserve the two-particle failure. The candidate runtime is not yet deployed, and no r3 dynamic or reserved result is claimed.

The standalone late-drip scalar helper was not applied: r2 case 05 additionally fails full-stop confirmation and loses three particles. `tools/refit_finish_level.py` now prepares a separate 4070-trained settled-level candidate using audited r0–r2 views as explicit training data; `tools/apply_finish_refit.py` archives the parent only after reviewed refit checks. No new fit or deployment is claimed yet. The original thirty reserved scenes remain unused. The full-stop trace is preserved in `development-r2-case05-full-stop-diagnostic.json`.
