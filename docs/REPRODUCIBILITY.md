# Reproducibility protocol

## Fixed sample and labels

One learning sample is one breast with its available ipsilateral CC/MLO views.
The two sigmoid targets are mass and suspicious calcification. The public split
assignment snapshot contains 10,000 breasts: 6,400 training, 1,600 validation
and 2,000 test samples. Examination grouping prevents breasts or views from the
same examination crossing split boundaries.

## Fixed image protocol

- MONOCHROME1/2 polarity standardization.
- Right-breast horizontal mirroring to a common chest-wall-left orientation.
- Otsu foreground with largest 8-connected component and hole filling.
- Conservative DICOM non-padding topology fallback for the four audited views.
- P1–P99 normalization inside the breast mask.
- 512 × 512 patches; stride 512 for training and 128 for evaluation.
- Inclusive minimum breast-mask coverage of 0.50.
- Gamma and contrast augmentation only; no geometric augmentation.

## Fixed optimization protocol

- Seed 2026.
- Batch size 4 with gradient accumulation 2, effective batch size 8.
- AdamW, learning rate `5e-5`, weight decay `1e-4`.
- Cosine decay to `1e-6`.
- Maximum 30 epochs with validation early stopping.
- Frozen Mammo-CLIP EfficientNet-B2 throughout training.

## Evaluation separation

`infer` runs the trained model once and writes a hash-guarded prediction
artifact. `evaluate` consumes that artifact and does not invoke the model
again. This separates model inference from metric implementation and ensures
all classification, localization and paired-comparison results use the same
predictions.

The formal localization protocol uses the 95th percentile response threshold
with a fixed lower bound of 0.5, 8-connected components, IoU 0.25, and no NMS,
minimum-area filter or maximum-box limit.

## Split snapshot

`artifacts/splits/vindr_breast_seed2026_assignments.csv` stores only public
VinDr identifiers, labels and split membership; machine-specific image paths
are intentionally excluded. Running `dagl-mil prepare` reconstructs a local
full manifest from the downloaded dataset and validates the locked counts.

