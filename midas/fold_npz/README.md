# fold_npz

Pre-computed 5-fold cross-validation splits for the MIDAS experiment.

## Contents

Each fold produces 3 NPZ files:

| File | Contents |
|------|----------|
| `active_dscope_fold{N}.npz` | Dermoscopy image paths + binary labels (train/val/test) |
| `passive_6in_fold{N}.npz` | 6-inch photography image paths (train/val/test) |
| `passive_1ft_fold{N}.npz` | 1-foot photography image paths (train/val/test) |

## Split sizes

| Split | Size |
|-------|------|
| Train | 423 lesions |
| Val | 105 lesions |
| Test | 132 lesions |

## Note

These files contain only image filenames and binary labels, no actual image data.
Download MIDAS images from Stanford AIMI and point scripts to your local image directory.
