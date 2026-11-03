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
| Current guarded visual model | `evidence/guard-A/best.pt` locally; Ubuntu `learning/A-guard-20261103-v1/best.pt` |
| Current model SHA256 | `f4ddb8b9f288ad2e5e8a9e6ac495f57a94c7413a0cdb4bbb040a89769bd36098` |
| Current native recordings/labels | `data/material-A-guard/`;19complete HDF5,76JSON sidecars, one aborted partial; checksum-verified |
| Current integrity manifest | Versioned `evidence/guard-A/recording-integrity.json`; local complete index `data/material-A-guard/backup-index.json` |
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
