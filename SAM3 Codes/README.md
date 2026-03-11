## Overview
Implement SAM3 to segment images and generate bounding boxes for detected species.

---

## Versions and Changes

### Code_v5.py
Creates bounding boxes and processes consecutive images taken in the same burst.

**Changes**
- Created bounding boxes around detected species. In earlier versions we cropped the image, which resulted in lower image quality.
- Added support for new JSON files:
  - `split_test_10_dynamic_burst_with_sam3_v3.json`
  - `split_train_80_dynamic_burst_with_sam3_v3.json`
  - `split_validation_10_dynamic_burst_with_sam3_v3.json`
- Using the new JSON files, consecutive images from the same burst can be passed to SAM3 so that bounding boxes are generated for all images, even if the detection occurs in only one image.
- In earlier versions, only the detection with the highest score and area was used. In this version, bounding boxes are created for **all detections**.