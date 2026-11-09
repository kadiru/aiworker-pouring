# Native-resolution rim localization

Completed November 9, 2026, on the guarded RTX 4070. **Localization passes its development checks; level estimation fails, and all nine reserved static tests remain unused.** The original robot controller and runtime sources are unchanged. This is a static, known-size perception result, not a resized autonomous-pouring result.

## Data and inference

The same 108 reviewed recordings provide 90 training and 18 validation episodes, spanning nine cup height/diameter combinations. No new recording or reserved image was generated. Localization samples the first twelve observations and final next observation: 1,170 training / 234 validation images. Simulator transforms supply offline labels and scoring only.

The r3 coarse CNN remains frozen at 384 pixels. Its RGB prediction selects a 256 × 256 crop from the native 1024 × 1024 image. A new 52,129-parameter CNN, with a 257-pixel receptive field, predicts a Gaussian rim-center heatmap; soft-argmax yields subpixel coordinates. Padding and crop origins are explicit. No inference crop uses a ground-truth position. A correction beyond 32 native pixels from the coarse prediction rejects the detection. Presence scoring remains the frozen coarse detector's score, so refinement cannot inflate an absent-cup score.

Training minimizes heatmap cross-entropy plus 0.02 times smooth-L1 coordinate loss, with equal size/episode sampling and AdamW. R4 uses integer translations and brightness augmentation. R5 adds continuous shifts, ±12° rotation, 0.9–1.1 scale and mild aspect changes. Both runs complete 160 epochs / 11,840 updates; neither passes the full 3 mm gate. R4's best validation attempt is epoch 20 at 3.0905 mm; r5 is 3.6523 mm. Full histories, failed artifacts and exact source versions remain in native `locator-rounds/r4` and `locator-rounds/r5`. Fitting takes 88.56 and 128.21 seconds respectively.

Residual diagnosis motivates a small image-coordinate calibration of retained r4 epoch 20. R6 fits six coefficients on one mean per training episode:

`delta_uv = [1, (u−512)/512, (v−512)/512] B`

`B = (XᵀX + diag(1e−8, 0.1, 0.1))⁻¹ Xᵀ(Y−U)`

This is a CUDA float64 ridge solve on the 4070; both CNNs stay frozen. The declared maximum correction is 2 native pixels, with rejection rather than clipping if exceeded. Largest measured correction is **0.8724 pixels**. The correction depends only on image-derived UV at inference. No RL is used.

## Accepted localization; rejected level fit

| Check | Measured | Required |
|---|---:|---:|
| Worst validation localization | 2.9033 mm | <3 mm |
| Worst training localization | 2.9766 mm | <3 mm |
| Validation mean localization | 1.4493 mm | Reported |
| Temporal confirmation | 108/108 | Every episode |
| Worst temporal localization | 2.8068 mm | <3 mm |
| Synthetic-absence score margin | 0.63219 | >0.015 |
| CPU/GPU native-pixel difference | 0.01763 px | <0.1 px |
| Batched/end-to-end inference difference | 0.00218 px | <0.01 px |

The margin is narrow: the worst training frame is only 0.0234 mm below the limit. All accuracy results above use consumed development data; no independent generalization claim follows. The separate level-stage camera path also confirms all 108 episodes, worst 2.8067 mm. Accepted locator checkpoint SHA: `a19e1ff32746e6541ecfad3e8b6e1ab88ab04a3d41a78c5fa274c98ea1717570`.

The existing static level fitter then runs 1,200 AdamW steps, fitting 5,025 parameters over frozen CNN/image features. It uses 990 training / 198 validation images from settled windows. No checkpoint meets all unchanged per-size gates. Best attempt is step 1080:

- Overall validation MAE: **2.0891 percentage points**; worst size MAE: **3.5548 points**, versus a **2-point per-size** limit.
- Five sizes fail that MAE limit: height/diameter 0.9/0.9, 0.9/1.1, 1/1, 1.1/0.9 and 1.1/1.
- Near-full maximum absolute error and maximum overestimate: **2.3683 points**, within their 3.5/2.5-point limits. Passing these alone is insufficient.
- CPU/GPU difference: **0.00177 percentage points**; parity passes.

The pipeline exits normally at `level_gate_failed`: **0/9 reserved tests**, no component selection and no active model replacement. The level artifact remains explicitly rejected. Parent checkpoint SHA `17717db90c9921aebba4f4ac67aded15927e7f07805a9b90406703bbafcb4428`, plan v5, protected historical evidence and runtime sources remain unchanged.

## Verification and provenance

Eight native CPU tests cover crop/padding coordinates, soft-argmax, image/label affine consistency, inference crop origin and correction rejection. All eighteen actual final validation overlays were inspected across three sheets; four initial error overlays were also inspected. This is selected-frame review, not continuous playback. The final queue is empty and recording inactive. The existing 4070-only guard, four-core quota, CPU set 16–23, low shares and 16 GiB limit remain; no foreign job was signaled or reconfigured.

Reports: [summary](../evidence/scale-A/rim-refit-summary.json), [r4](../evidence/scale-A/locator-r4.json), [r5](../evidence/scale-A/locator-r5.json), [r6](../evidence/scale-A/locator-r6.json), [level](../evidence/scale-A/level-r6.json), [closure](../evidence/scale-A/rim-pipeline-r6-completion.json), [manual review](../evidence/scale-A/rim-r6-manual-review.json).

Native artifacts live under `/workspace/learning/A-scale-20261109-v1` in owned container `0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1`. Host path: `/home/kadir/aiworker-pouring-4070/learning/A-scale-20261109-v1`. Weights, camera images and HDF5 remain external to Git. Each new round was declared before its native fit and the previous failed round archived first. No threshold or held-out identity changed.

Completed host commands were `python3 .../run_scale_rim_pipeline.py` for r4, then `--round r5` and `--round r6`. These are exclusive-output commands and must not be blindly rerun. The native CPU implementation check is:

```sh
ssh kfuUbuntu 'docker exec -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1 nice -n 10 /isaac-sim/python.sh /workspace/learning/A-scale-20261109-v1/test_scale_rim_refiner.py -v'
```

Next: preserve accepted localization and diagnose level generalization across sizes using existing training/validation records. Keep all nine reserved scenes unused until every component gate passes. Higher-fill (≥94%), held-bottle/dynamic perception, motion clearance and bottle-size grasp validation still precede resized autonomous pouring.
