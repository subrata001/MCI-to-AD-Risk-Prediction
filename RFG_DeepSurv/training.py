# -------------------------------------------------------------------- #
# Edit the CSV paths below, then run python training.py
# -------------------------------------------------------------------- #

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch.optim as optim
from pycox.models import CoxPH
from lifelines.utils import concordance_index
from data import load_data
from model import RFG_DeepSurv

ADNI_CSV = "ADNI_sMCI_pMCI_Clincal_MRI_Embeddings.csv"
OASIS_CSV = "OASIS3_sMCI_pMCI_Clinical_MRI_Embeddings.csv"
OUTPUT_DIR = Path("outputs")
N_PCA_COMPONENTS = 20


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    data = load_data(ADNI_CSV, OASIS_CSV, N_PCA_COMPONENTS)
    
    X_train, X_internal = data["X_train"], data["X_internal"]
    
    y_train, y_internal = data["y_train"], data["y_internal"]
    
    durations_train, events_train = y_train
    
    durations_internal, events_internal = y_internal
    
    joblib.dump(data["preprocessing_bundle"], OUTPUT_DIR / f"deepsurv_preprocessing_bundle_PCA{N_PCA_COMPONENTS}.joblib", compress=3)
    
    net = RFG_DeepSurv(in_features=X_train.shape[1], hidden_nodes=[64, 32], dropout=0.2)
    
    optimizer = optim.SGD(net.parameters(), lr=0.0004, momentum=0.9, weight_decay=1e-4)
    
    deepsurv_model = CoxPH(net, optimizer)
    
    batch_size, epochs, patience = 32, 500, 30
    
    history = []
    best_val_loss, best_epoch, patience_counter = np.inf, 0, 0
    best_model_path = OUTPUT_DIR / "RFG_DeepSurv_best.pt"
    for epoch in range(1, epochs + 1):

        # Train for one epoch
        log = deepsurv_model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=1,
            verbose=False,
            val_data=(X_internal, y_internal)
        )

        log_df = log.to_pandas()
        last_row = log_df.iloc[-1]

        # Handle different pycox/torchtuples column names
    
        if "train_loss" in last_row.index:
            train_loss = float(last_row["train_loss"])
        elif "loss" in last_row.index:
            train_loss = float(last_row["loss"])
        else:
            train_loss = np.nan

        if "val_loss" in last_row.index:
            val_loss = float(last_row["val_loss"])
        else:
            val_loss = np.nan

        # Calculate train and validation C-index

        train_risk = deepsurv_model.predict(X_train).reshape(-1)
        val_risk = deepsurv_model.predict(X_internal).reshape(-1)

        
        # DeepSurv risk: higher value = higher hazard = shorter survival
        # lifelines concordance_index assumes higher score = longer survival
        # Therefore, we use negative risk
        
        train_cindex = concordance_index(
            durations_train,
            -train_risk,
            events_train
        )

        val_cindex = concordance_index(
            durations_internal,
            -val_risk,
            events_internal
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_cindex": train_cindex,
            "val_cindex": val_cindex,
        })

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.3f} | "
            f"val_loss={val_loss:.3f} | "
            f"train_cindex={train_cindex:.3f} | "
            f"val_cindex={val_cindex:.3f}"
         )

        # Early stopping based on validation loss

        if val_loss < best_val_loss :
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            deepsurv_model.save_model_weights(str(best_model_path))

        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            print(f"Best epoch: {best_epoch}, best val_loss: {best_val_loss:.4f}")
            break
            
    #pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)

if __name__ == "__main__":
    main()
