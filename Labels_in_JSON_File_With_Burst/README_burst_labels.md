# Methodology (Dynamic Burst Construction)

### 1. Input Data Preparation

- Input: The full set of labels with corresponding valid directories of the dataset.

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

Dynamic burst segmentation was performed on temporally ordered frames within each camera stream. For each pair of consecutive frames, we computed two signals: (i)
temporal gap delta_t in seconds from parsed timestamps, and (ii) visual change using Hamming distance between 64-bit aHash descriptors. A new burst was initiated when
any one of the following conditions was satisfied:

1. Location change
    If the camera location identifier changed between two consecutive frames, the sequence was forcibly split. This prevents cross-camera contamination within the same
    burst.
2. Non-monotonic timestamp (delta_t < 0)
    If the next frame appeared earlier than the previous frame in time (clock mismatch, filename anomaly, or ordering issue), a new burst was started to maintain
    temporal consistency.
3. Hard temporal discontinuity (delta_t > 90 s)
    If inter-frame time exceeded 90 seconds, the pair was treated as different events regardless of visual similarity.
4. Joint temporal-visual discontinuity (delta_t > 30 s and Hamming > 20)
    For medium time gaps, splitting required both temporal separation and moderate visual change. This avoids over-splitting when scenes are stable despite moderate
    timestamp gaps.
5. Large visual discontinuity (Hamming > 28)
    If visual change was very large, a new burst was created even when frames were close in time, capturing abrupt scene transitions (e.g., animal enters/exits, major
    pose/background change).

The fixed hyperparameters used in all v3 experiments were:

- hard_gap_sec = 90
- soft_gap_sec = 30
- hash_split_threshold = 20
- large_change_threshold = 28

Operationally, soft_gap_sec and hash_split_threshold define the joint rule, while large_change_threshold acts as an override for strong visual transitions. This
design yields variable-length bursts that are constrained by both temporal continuity and image-content continuity.

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

### 6. Reproducibility

- Deterministic components:

1. Fixed thresholds.
2. Deterministic sorting before segmentation.
3. Data with same burst ID should not be splitted across train, valid and test sets of data (i.e., It should be burst proof.).

“The dynamic event-level bursts are constructed using a hybrid temporal-visual segmentation algorithm. Consecutive frames were grouped by location/date and segmented using
filename-derived inter-frame time gaps and perceptual frame change measured by 64-bit average-hash Hamming distance. Burst labels were assigned by within-burst
majority vote, and data were split at burst level to prevent cross-split leakage of temporally adjacent frames.”

