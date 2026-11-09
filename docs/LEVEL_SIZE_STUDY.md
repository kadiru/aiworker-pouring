# Liquid level across cup sizes

Study: `A-scale-20261109-v1`, November 9, 2026. This continues the [rim localization study](RIM_LOCALIZATION.md). It concerns stationary RGB perception with known cup dimensions, not resized autonomous pouring or visual size estimation. The active robot checkpoint and runtime sources are retained.

## Method and acceptance

The output is liquid surface height as a fraction of usable cup height, not nominal volume. Supervision is the measured settled particle surface in the final second of each simulation recording. All frames from an episode stay in its declared split. Nine combinations use height and diameter scales 0.9, 1.0 and 1.1. Bottle size, material A and fluid physics are unchanged.

The frozen native-resolution rim estimator supplies image-localized XY for the level crop. Declared cup dimensions and the calibrated camera are fixture inputs. Neither simulator XY nor the ground-truth liquid level is supplied to inference. The trainable model uses a normalized 96-pixel RGB crop and a small context correction conditioned on height, diameter, aspect ratio, predicted XY and the CNN level. It has **58,770 trainable parameters**. Training minimizes equal-size/equal-episode weighted mean squared error with AdamW, modest crop/brightness augmentation and held-out development checkpoint selection. This is supervised regression, not reinforcement learning.

Every original gate remains: localization maximum below 3 mm; absence-score margin above 0.015; **each size's training and validation level MAE below 2 percentage points**; near-full maximum absolute error below 3.5 points and maximum overestimate below 2.5 points; CPU/GPU level difference below 0.0001 fraction. Temporal localization and shared inference-path checks also apply. No aggregate average can override a failing size.

The nine reserved scenes must remain unused until all gates pass and the selected components are frozen. They are once-only **stationary near-full perception tests**, not nine autonomous pours. Independent color, higher-fill (at least 94%), held-bottle/dynamic perception, motion clearance and task evaluation remain separate requirements before resized autonomy.

## Rejected development fits retained

All fits ran in the existing guarded 4070-only container. The 108-record dataset had 90 training and 18 validation episodes. These validation views were reused for development; only the nine reserved identities can supply the later independent check.

| Revision | Change | Worst-size validation MAE | Near-full max absolute error | Result |
|---|---|---:|---:|---|
| r7 | Frozen CNN/color features with declared geometry; 44 ridge/RBF candidates | 2.6256 points | 3.6708 points | Failed |
| r8 | Native 192-pixel crop, red surface quantiles and projected physical-height lookup; 22 candidates | 2.9157 points | 3.4835 points | Failed |
| r9 | Fine-tune CNN and context correction, 240 epochs / 3,840 updates; best epoch 110 | 3.0610 points | 1.2498 points | Failed |

R7 and r8 selected 91-coefficient RBF readouts as their best rejected attempts. Fit times were 2.10 and 1.25 seconds, excluding dataset extraction. R9 fitting took 32.18 seconds (61.09 seconds including extraction). All three passed CPU/GPU parity, but no checkpoint passed every accuracy gate; no reserved scenes were started by those pipelines. Full reports and completion records are in `evidence/scale-A/level-r7.json` through `level-r9.json` and the corresponding `level-pipeline-*-completion.json` files. Native rejected weights remain archived, outside Git.

R9 improves near-full discrimination but retains low-fill errors: training fits are much better than validation, with the largest errors between sparsely sampled levels. This is a development diagnosis, not proof that data coverage is the sole cause.

## Additional training coverage

Plan v6 adds **36 training-only static recordings**, four existing training layouts per size at a 0.35 initial seed-height cut. The cut defines the initial condition; it is not used as the measured target. All original 108 episode specifications, 18 validation identities, nine reserved identities and thresholds remain identical. Total planned collection is **144 episodes: 126 training / 18 validation**. Each size now has 14 training episodes.

All **144/144** recordings pass native HDF5 integrity and final-second stable-label checks. The **36/36** new scenes have zero measured loss and measured height fractions **32.4027–34.1185%**. All six sheets containing their 36 final cup-detail panels were actually opened and inspected: rims and red surfaces are visible, with no obvious bottle occlusion or missing liquid. This is selected-frame review, not continuous playback. The previous 108 reviews carry forward only after identical recording SHA checks.

The unchanged rim checkpoint passes revalidation on all 144 episodes and all 1,872 sampled frames. Maximum error is **2.994676 mm training / 2.903344 mm validation**; all 144 temporal confirmations pass. The training margin is only **0.005324 mm**, so this is fragile development acceptance, not evidence of robust new-layout localization. No rim weights were refitted. The declared r10 fit continues from the archived rejected r9 checkpoint, with 240 epochs and 5,280 expected AdamW updates. An inherited failed checkpoint is an initialization only, never an accepted model.

R10 **passes every development gate**. The 240-epoch fit completed **5,280 AdamW updates** in **40.23 seconds** (72.64 seconds with extraction), selecting epoch 110. It uses 1,386 training / 198 validation images from the 126/18 whole-episode split. Validation overall MAE is **0.9759 percentage points**, worst-size MAE **1.3329 points**, and near-full maximum absolute error/overestimate **1.0643 points**. CPU/GPU maximum difference is **0.00002259 fraction**, below 0.0001. All 144 temporal localizations also pass immediately before level extraction. The accepted weights are frozen as study-only components; the active robot checkpoint is unchanged.

## Independent static result

After every development gate passed, the selected component hashes were frozen and all **nine reserved scenes ran exactly once**. **9/9 pass** the unchanged per-case localization and level requirements. Mean level error is **0.6578 percentage points**, maximum absolute error **1.4194 points**, maximum overestimate **1.0636 points**, and maximum localization error **2.4006 mm**. Actual settled labels span **90.3812–91.1482%** of usable height. All nine recordings are valid and stable, with zero measured loss. Both review sheets containing all nine final cup views were actually inspected; rims and red liquid surfaces are visible without obvious occlusion or capture corruption.

The retained independent color readout has a **9.3882-point maximum error** on these views. It was a predeclared diagnostic, not a component-selection gate, and remains unaccepted for the resized task. The nine tests also do not establish low-fill independent accuracy, >=94% fill accuracy, moving/held-bottle perception or autonomous pouring. Development validation was reused across candidate searches. No test result was used to refit or retry a scene, and no active controller was replaced.

The final queue is empty, recording is inactive, and all protected parent files and runtime source hashes match their pre-study identities. The accepted level checkpoint is `e218aa9e25dde7bbf453ae36d0b45eb6192b196d4a26c02d451ad8c9cc15b6a0`. [Verified summary](../evidence/scale-A/level-size-summary.json), [fit](../evidence/scale-A/level-r10.json), [independent scores](../evidence/scale-A/static-holdout-evaluation.json), [camera review](../evidence/scale-A/holdout-r0-manual-review.json), [guarded completion](../evidence/scale-A/level-pipeline-r10-completion.json).

## Verification and continuation

Eleven focused checks passed: four readout/geometry-feature invariants, three projected-surface checks, two CNN/context/shared-path checks, and two training-supplement split-preservation checks. The first nine ran in the native CPU environment; the two plan checks ran locally. The actual revalidation, fit, inference parity and once-only evaluation supply the simulation evidence; synthetic unit-test inputs are not task results.

Recheck the local source-controlled evidence with:

```sh
python3 tools/summarize_scale_level.py
```

The completed native launch was `python3 /home/kadir/aiworker-pouring-4070/learning/A-scale-20261109-v1/run_scale_level_pipeline.py --round r10`. Its existing output directories deliberately prevent rerunning the consumed reserved IDs. Do not remove those guards or overwrite the frozen artifacts. Native restoration requires the external models, recordings and Isaac environment listed below.

Next: calibrate the independent color check and collect newly declared low-fill, >=94%-fill and held-bottle/moving views before integration. Then verify all-size grasp/clearance/side-return behavior and fresh autonomous task outcomes. Keep bottle scaling disabled until its separate geometry/grasp checks pass. The completed nine-test result remains frozen.

## Artifacts and isolation

Native data and model artifacts: `/workspace/learning/A-scale-20261109-v1` in `aiworker_pouring_4070`; host `/home/kadir/aiworker-pouring-4070/learning/A-scale-20261109-v1`. New raw recordings remain under `/workspace/outputs/fountain/20261109T170659Z`. Selected review sheets are retained locally under `evidence/scale-A/collect-r2-camera-review/`; images, HDF5 files and weights are outside Git.

The immutable owned container remains limited to the RTX 4070, four CPU cores of quota on CPUs 16–23, CPU shares 256 and 16 GiB memory/swap. The existing priority guard stops only this container if another 4070 workload appears. This task did not signal, reconfigure or stop the 4090 workload. Shared-host resource use is bounded, not zero.

The pre-existing 4090 PID 4172782 was present in earlier snapshots and absent by the 15:04 PST collection snapshot; a read-only `ps` check confirmed that PID had exited. Its exit reason was not determined. This task did not signal that PID or change its configuration; snapshots do not establish uninterrupted execution of other jobs.

Active parent checkpoint SHA: `17717db90c9921aebba4f4ac67aded15927e7f07805a9b90406703bbafcb4428`. Frozen rim checkpoint SHA: `a19e1ff32746e6541ecfad3e8b6e1ab88ab04a3d41a78c5fa274c98ea1717570`. Plan v6 SHA: `4efcc93638e4abc10a6222712470f395aa71a5812e117373d6a1693c6364bf47`.
