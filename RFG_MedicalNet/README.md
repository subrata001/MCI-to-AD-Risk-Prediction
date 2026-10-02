# RFG-MedicalNet training

Three Python files:
- train.py: CSV reading, DataLoaders, training/evaluation functions, optimizer, training loop, and output saving.
- data.py: original BrainAgeDataset, unchanged.
- model.py: residual gating, age regressor, and MedicalNet initialization.

Edit base_dir, resnet_path, pretrained_path, and output_dir inside main() in train.py, then run:

```bash
python train.py
```

Dependencies: torch, numpy, pandas, nibabel, tqdm, scikit-learn, scipy. Use the versions from your original experiment. Also supply your original MedicalNet models/resnet.py and pretrained ResNet-10 checkpoint. Python 3.9+ is needed for removeprefix; torch must support weights_only=True.

CSV filenames, columns, dataset construction, and DataLoader settings match the uploaded script. CSVs must contain OASISID, preprocessed_image_path, and age_label. Image paths are passed directly to nibabel, as in the original; relative paths are relative to the working directory. Input images must already be preprocessed. No new CSV/shape/partition validation is added. Ensure original CN eligibility and subject-disjoint splits yourself.

The uploaded script's undefined gating variables and incorrect model class name are corrected as in the earlier package. Architecture, training defaults and original numpy nonfinite handling are retained. The gate dropout argument remains unused. MedicalNet weight loading reports key mismatches and requires compatible pretrained backbone weights instead of silently training from scratch.

The best validation-MAE checkpoint is saved to outputs, alongside history and held-out test predictions/metrics. Test evaluation and CSV saving were added to the original training script. Bias correction and embedding extraction remain separate pending source code.

Syntax and structural checks passed. Full training and checkpoint compatibility were not tested because torch, nibabel, real data, MedicalNet source, and weights were unavailable. Check inferred model corrections against the original working experiment before publication. Do not upload private data, local manifests, or weights unintentionally. The final repository license and original environment versions remain to be supplied.
