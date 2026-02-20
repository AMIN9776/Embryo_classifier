# Embryo Development Classification Pipeline

Two-stage PyTorch pipeline for classifying 16-stage embryo development from multi-focal time-lapse images.

## Project Structure

```
embryo_project/
├── config.yaml                  # Master config — controls everything
├── dataset.py                   # Data loading, augmentation, preprocessing
├── models.py                    # All model components
├── losses.py                    # All loss functions
├── train_pretrain.py            # Self-supervised pretraining loop
├── train_finetune.py            # Supervised fine-tuning loop
├── utils.py                     # Logging, checkpointing, metrics
├── preprocessing_transforms.py  # (your file) preprocessing transforms
└── requirements.txt
```

## Expected Data Structure

```
/data/embryo/
├── embryo_dataset_F1/
│   ├── Patient001/
│   │   ├── t001.png
│   │   ├── t002.png
│   │   └── ...
│   └── Patient002/
│       └── ...
├── embryo_dataset_F2/ ... (same patient/frame structure)
├── ...
├── embryo_dataset_F7/
├── annotations.csv              # patient,stage,start_frame,end_frame
├── train_patients.json          # {"patients": ["Patient001", ...]}
└── test_patients.json           # {"patients": ["Patient099", ...]}
```

### Annotation CSV format
```csv
patient,stage,start_frame,end_frame
Patient001,1,1,5
Patient001,2,6,12
Patient001,3,13,18
...
```

### Patient JSON format
```json
{"patients": ["Patient001", "Patient002", "Patient003"]}
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Edit config.yaml
Update `data.data_root`, `data.train_json`, `data.test_json`, and `data.annotation_csv`.

### 3. Run pretraining
```bash
python train_pretrain.py --config config.yaml
# Resume from checkpoint:
python train_pretrain.py --config config.yaml --resume
```

### 4. Run fine-tuning
```bash
python train_finetune.py --config config.yaml \
    --pretrain_checkpoint ./outputs/pretrain/best_model.pth
```

### 5. Cross-validation
```bash
python train_finetune.py --config config.yaml --cv
# Or a specific fold:
python train_finetune.py --config config.yaml --fold 2
```

## Architecture Overview

### Stage 1: Self-Supervised Pretraining

```
[7 focal planes] → Shared ResNet/ViT Encoder
                        ↓
          ┌─────────────┼─────────────┐
          ↓             ↓             ↓
  [Contrastive]      [MAE]       [Temporal]
  Cross-focal        Mask &       Gap bin
  InfoNCE           Reconstruct   prediction
```

**Three objectives:**
- **Cross-focal InfoNCE**: Same embryo/time across focal planes = positives
- **MAE reconstruction**: Mask 75% of patches, reconstruct best-focus plane
- **Temporal SSL**: Predict time gap bin between two timepoints (Δ∈{≤1,2,3-4,5-8,≥9})

### Stage 2: Supervised Fine-Tuning

```
[7 focal planes] → Encoder (partially/fully unfrozen)
                        ↓
              [Focal Attention Fusion]
                        ↓
              [Temporal GRU/Transformer]
                        ↓
     [Soft MoE: 3 Expert Heads + Router]
      Expert A        Expert B      Expert C
    (Early 1-4)     (Mid 5-8)    (Late 9-16)
                        ↓
                [16-class output]
             + Auxiliary: cell count
             + Auxiliary: blastocyst score
             + Monotonicity loss
```

**3-phase progressive unfreezing:**
| Phase | Epochs | Encoder | LR |
|-------|--------|---------|-----|
| 1 | 0–9 | Frozen | head LR |
| 2 | 10–24 | Last 2 blocks unfrozen | encoder LR × 0.1 |
| 3 | 25–49 | Full | encoder LR × 0.1 × layer_decay^depth |

## Key Config Options

| Config key | Description |
|---|---|
| `preprocessing.method` | `null`, `clahe`, `canny`, `minmax`, `hsv`, `hsv_normal`, `high_pass`, `low_pass`, `wavelet` |
| `model.backbone` | `resnet18`, `resnet34`, `resnet50`, `vit_small`, `vit_base` |
| `model.fusion.method` | `mean`, `max`, `attention`, `cross_attention` |
| `model.classifier.use_moe` | Enable/disable Mixture of Experts |
| `model.classifier.use_temporal_gru` | Enable temporal GRU in fine-tuning |
| `pretrain.mae.reconstruction_target` | `same` or `best_focus` |
| `pretrain.temporal_ssl.task` | `gap_prediction` or `order_prediction` |
| `data.use_cross_validation` | Enable K-fold CV |
| `data.annotation_format` | `csv` or `folder` |

## Output Structure

```
outputs/
├── pretrain/
│   ├── best_model.pth
│   ├── last_checkpoint.pth
│   ├── epoch_XXXX.pth
│   └── config.yaml
└── finetune/
    ├── best_model.pth
    ├── last_checkpoint.pth
    └── config.yaml

logs/
├── pretrain_TIMESTAMP/
│   ├── training.log
│   ├── tensorboard/
│   └── csv/
│       ├── train_step.csv
│       └── train_epoch.csv
└── finetune_TIMESTAMP/
    ├── training.log
    ├── tensorboard/
    └── csv/
        ├── finetune_train_epoch.csv
        ├── finetune_val_epoch.csv
        └── test_final.csv
```

## Adding Custom Preprocessing

Your `preprocessing_transforms.py` is automatically imported. To use a method:
```yaml
# config.yaml
preprocessing:
  method: "clahe"   # or: minmax, canny, laplacian, hsv, hsv_normal, high_pass, low_pass, wavelet
```

To add new transforms, register them in `preprocessing_transforms.py`:
```python
TRANSFORM_REGISTRY['my_method'] = MyTransformClass
```
Then set `preprocessing.method: "my_method"` in config.

## Memory Tips (RTX 3090 — 24GB VRAM)

- `resnet50` + 7 focal planes + batch 64: ~18GB → use `use_amp: true`
- `vit_small` + MAE: reduce `batch_size` to 32 and enable `gradient_checkpointing: true`
- Reduce `num_workers` if you hit CPU bottlenecks
- Use `image_size: [128, 128]` for fast prototyping
