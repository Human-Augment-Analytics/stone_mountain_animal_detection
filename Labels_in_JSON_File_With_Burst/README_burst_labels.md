• Methodology (Dynamic Burst Construction)

### 1. Input Data Preparation

- Source annotations were loaded from three JSON files:

1. split_train_80_with_sam3.json
2. split_validation_10_with_sam3.json
3. split_test_unseen_10_with_sam3.json

- Each record includes image paths, metadata, and labels (correct_label, location, capture_date, night_time).
- To avoid duplicate frame variants, one record per original frame was retained using file_path_original (or file_path fallback), prioritizing image_variant ==
"original" when duplicates existed.
- SAM3 companion paths (file_path_sam3) were preserved for downstream multimodal training.

### 2. Temporal Signal Extraction

- A timestamp was parsed from the filename pattern _<YYYYMMDD>_<HHMMSS>__.
- Frames were organized by (location, capture_date) and sorted by:

1. Parsed timestamp.
2. File path (tie-breaker).

### 3. Pixel-Based Change Signal

- For each frame, an average hash (aHash) was computed:

1. Convert image to grayscale.
2. Resize to 8 x 8.
3. Compute global mean pixel intensity.
4. Set bit 1 if pixel >= mean, else 0.
5. Produce a 64-bit hash.

- Between consecutive frames, visual change was measured by Hamming distance between hashes.

### 4. Dynamic Burst Boundary Rules

A new burst was started if any of the following held between consecutive frames:

1. location changed.
2. Timestamp moved backwards (delta_t < 0).
3. Hard temporal discontinuity: delta_t > 90 s.
4. Joint temporal+visual discontinuity: delta_t > 30 s and Hamming > 20.
5. Large visual discontinuity regardless of time: Hamming > 28.

These thresholds were fixed:

- hard_gap_sec = 90
- soft_gap_sec = 30
- hash_split_threshold = 20
- large_change_threshold = 28

### 5. Burst Label Assignment

For each dynamic burst:

- correct_label: majority vote of frame-level labels in the burst.
- night_time: mode of frame-level day/night flags.
- label_distribution: full class frequency map retained for traceability.
- num_frames: number of frames in the burst.
- frames: list containing file_path, file_path_original, file_path_sam3, image_variant, capture_date.

Burst IDs were generated as:

- {location}_{capture_date}_dynburst_{6-digit-index}

- Splitting was performed at burst level (not frame level), preventing leakage of related frames across splits.
- Ratios: 80% / 10% / 10%.
- Random seed: 42.
- Bursts were shuffled, then assigned to train/validation/test.

### 7. Final v3 Output Statistics

Generated files:

1. split_train_80_dynamic_burst_with_sam3_v3.json
2. split_validation_10_dynamic_burst_with_sam3_v3.json
3. split_test_10_dynamic_burst_with_sam3_v3.json

Observed counts:

- Train: 1966 bursts, 10152 frames
- Validation: 246 bursts, 1264 frames
- Test: 245 bursts, 1322 frames
- Total: 2457 bursts
- Mean burst length: 5.184 frames

### 8. Reproducibility

- Deterministic components:

1. Fixed thresholds.
2. Fixed seed (42) for split assignment.
3. Deterministic sorting before segmentation.

### 9. Suggested Paper Wording (Short)

“The dynamic event-level bursts are constructed using a hybrid temporal-visual segmentation algorithm. Consecutive frames were grouped by location/date and segmented using
filename-derived inter-frame time gaps and perceptual frame change measured by 64-bit average-hash Hamming distance. Burst labels were assigned by within-burst
majority vote, and data were split at burst level (80/10/10) to prevent cross-split leakage of temporally adjacent frames.”

