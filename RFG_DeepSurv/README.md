# RFG-DeepSurv training

Three Python files:
- `data.py`: loads the ADNI/OASIS3 CSVs, harmonizes columns, removes incomplete rows, splits ADNI, and fits clinical scaling, embedding scaling and PCA on ADNI training rows only.
- `model.py`: residual gating and the survival log-risk network.
- `training.py`: paths, optimizer, training, early stopping and saving.

Install dependencies: `pip install torch numpy pandas scikit-learn joblib pycox torchtuples lifelines`.
Edit the two CSV paths in `training.py`, then run `python training.py`.

## CSV inputs
Clinical columns: age_for_cox, sex_male (1=male, 0=female), EDUC, MMSE, CDRSUM, BrainAGE_corrected. Alternative column names supported by the original harmonization function remain supported. BrainAGE_corrected is the corrected brain-age gap, not predicted brain age. MRI embedding columns start with `emb_`; their ADNI CSV column order is retained and applied to OASIS3. Provide matching columns in both CSVs. Images and brain-age weights are not loaded here.

Outcome: time_to_event_years (or time_to_event_days, converted by 365.25) and event (1=conversion, 0=censored). For censored rows, duration means time to last follow-up. Alternatively progression_group maps pMCI_to_AD to 1 and sMCI to 0. Rows with nonpositive duration or missing required values are excluded.

## Training and saved outputs
Original settings retained: stratified ADNI 80/20 split, random_state=42, PCA-20, hidden layers 64/32, SELU, dropout=0.2, SGD learning rate 0.0004, momentum=0.9, weight decay=0.0001, batch size 32, maximum 500 epochs, patience 30.

The ADNI 20% partition is used for validation and early stopping, as in the upload. It is therefore not an untouched internal test set. There is no separate validation partition in the supplied script. Splitting is row based: ensure participants do not occur in multiple partitions if CSVs contain repeated scans. OASIS3 is preprocessed but never used for checkpoint selection or per-epoch monitoring.

`outputs/` contains best network weights, preprocessing bundle (including exact feature order and split indices), training history, and training-derived baseline hazards after restoring the best weights. Retain all these artifacts for the later single-patient prediction stage. No individualized prediction or final survival evaluation is included here.

## Corrections to the uploaded script
- Added the missing torch.nn import, corrected undefined gating dimension and model names, unpacked the gating tuple before the risk network, and corrected the malformed bundle path.
- Removed the first scaling block whose output was overwritten by the clinical/PCA pipeline, and redundant imports.
- Filtered positive duration using harmonized years so years-only CSVs work.
- Removed external C-index checkpoint selection (>0.695) and external per-epoch monitoring; best checkpoint uses validation loss only.
- Added history saving and baseline-hazard saving for later survival predictions.

These corrections require comparison with the working code used for the manuscript. The uploaded file is not executable as supplied. Syntax and synthetic CSV preprocessing were checked; full training, checkpoint compatibility and agreement with manuscript results remain untested because PyTorch, pycox, torchtuples, lifelines and study data are unavailable in this environment. The gate retains the supplied SELU/sigmoid residual formula and parameter names; identity with the working MedicalNet gate has not been verified. Gating dropout remains unused, matching the supplied gate.
