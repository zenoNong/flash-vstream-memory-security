# Research Changelog — Flash-VStream Memory Security

## [0.1.0] — 2026-09-17

### Initial Setup
- Cloned Flash-VStream at reference commit `8f6bde2f397f4846505df4d03364d97617fc02ee`
- Created research project structure
- Computed and recorded SHA-256 hashes for all key source files

### Audit Findings
- **Candidate A** (return-arity inconsistency): Confirmed. `weighted_kmeans_ordered_feature()` returns 3 values when T<=T0, 4 values when T>T0. Not triggered in normal pipeline because `temporal_compress()` has an early-return guard. Documented, no fix needed.
- **Candidate B** (cosine retrieval bug): Confirmed. `spatial_enhance()` uses `argmin` for both Euclidean and cosine paths. For cosine similarity, should use `argmax`. Only affects non-default `klarge_retrieve_cos` method. Fixed in research DAM adapter.
- **Candidate C** (timestamp overwrite): Confirmed. Weighted centroid timestamps computed at L265-276 are overwritten by arithmetic-mean timestamps at L278. Dead code. Documented, no fix applied to reference.
- **Candidate D** (batch_size=1): Confirmed. Documented as known limitation.
- **Candidate E** (numerical stability): Identified potential NaN from negative sqrt in Euclidean distance. Added tests.

### Additional Observations
- F1: `temporal_compress()` unpacks 4 values but some compression methods return 3. Only works because default method (`kmeans_ordered`) returns 4.
- F2: `temporal_pool()` requires `xdim == 3 * 2 * 14 * 14` (Qwen2VL-specific).
- F3: Debug print in `efficient_euclidean_distance()` fires on every call.
- F4: `fast_weighted_kmeans_ordered_feature()` is functionally identical to `weighted_kmeans_ordered_feature()` minus dead code.
- F5: Realtime variant has different `temporal_compress()` call signature (streaming interface).

### Created
- Research adapters (CSM, DAM)
- Memory state instrumentation (recorder, hooks)
- Research metrics (CSM retention, DAM retrieval, representation similarity, memory occupancy)
- Intervention framework
- Test suite (unit, integration, regression)
- Colab environment files
- Experiment 0 configuration
- Reference manifest with file hashes
