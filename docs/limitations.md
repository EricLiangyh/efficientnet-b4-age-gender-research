# Limitations and research roadmap

## Established evidence

- Historical CSV logs support the reported final-epoch validation values.
- Backbone and augmentation/loss/alignment variants have separate archived records.
- Model checkpoint checks establish file integrity, structural compatibility and finite numerical execution only.

## Unresolved limitations

- Original sample order, identity grouping and partition membership were not retained.
- Historical results are single-seed observations. Small differences do not establish statistical significance.
- Dataset merge quality, cross-source duplication and demographic representativeness remain unaudited.
- Small illustrative robustness examples cannot establish population-level robustness. Signed single-image error is not MAE.
- Historical detector timings without CUDA synchronization are not reliable model benchmarks. No 30 FPS end-to-end claim is made here.
- Local IMDB-derived filtering and label mapping need an independently documented reconstruction. Unknown labels must not silently become class 0.
- A shared backbone is an established multi-task design, not a novel backbone architecture.

## Next research experiments

1. Recover source identity metadata; deduplicate across sources and freeze an identity-disjoint protocol.
2. Train all comparisons under matching data, resolution, budget and seed conditions; repeat multiple seeds.
3. Evaluate one untouched external test set after selecting configuration on validation only.
4. Report age-bin/source subgroup metrics, classification F1, confidence intervals and failure cases.
5. Run controlled, multi-image perturbation tests for illumination, occlusion and resolution.
6. Measure synchronized model latency at explicit batch sizes and hardware; measure full video pipelines separately.

Dataset and checkpoint use remain subject to original sources' terms. Public redistribution and any deployment require a separate review of those terms and authorized imagery.
