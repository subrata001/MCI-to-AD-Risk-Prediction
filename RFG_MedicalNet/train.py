import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from data import BrainAgeDataset
from model import build_model

def training(model, loader, optimizer, criterion, device):
    model.train()

    losses = []
    preds = []
    targets = []

    for imgs, ages in tqdm(loader, desc="Training", leave=False):
        imgs = imgs.to(device)
        ages = ages.to(device)

        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, ages)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds.extend(outputs.detach().cpu().numpy())
        targets.extend(ages.detach().cpu().numpy())

    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))

    return np.mean(losses), mae, rmse


def evaluate(model, loader, criterion, device):
    model.eval()

    losses = []
    preds = []
    targets = []

    with torch.no_grad():
        for imgs, ages in tqdm(loader, desc="Evaluating", leave=False):
            imgs = imgs.to(device)
            ages = ages.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, ages)

            losses.append(loss.item())
            preds.extend(outputs.detach().cpu().numpy())
            targets.extend(ages.detach().cpu().numpy())

    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)

    try:
        r, p = pearsonr(targets, preds)
    except ValueError:
        r, p = np.nan, np.nan

    return np.mean(losses), mae, rmse, r2, r, p, np.array(targets), np.array(preds)


def main():
    print(torch.__version__)
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Edit these four paths for your computer.
    base_dir = "./data"
    resnet_path = "./MedicalNet/models/resnet.py"
    pretrained_path = "./MedicalNet/pretrain/resnet_10_23dataset.pth"
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Edit according to the dataset
    train_csv = f"{base_dir}/OASIS3_BrainAge_train.csv"
    val_csv   = f"{base_dir}/OASIS3_BrainAge_val.csv"
    test_csv  = f"{base_dir}/OASIS3_BrainAge_test.csv"

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Edit according to the dataset
    print("Train:", len(train_df), "subjects:", train_df["OASISID"].nunique())
    print("Val:", len(val_df), "subjects:", val_df["OASISID"].nunique())
    print("Test:", len(test_df), "subjects:", test_df["OASISID"].nunique())

    batch_size = 4

    train_dataset = BrainAgeDataset(train_df)
    val_dataset = BrainAgeDataset(val_df)
    test_dataset = BrainAgeDataset(test_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(len(train_dataset), len(val_dataset), len(test_dataset))

    model = build_model(resnet_path, pretrained_path, device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=50
    )

    save_path = os.path.join(output_dir, "RFG_MedicalNet.pth")

    num_epochs = 100
    best_val_mae = np.inf

    history = []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_mae, train_rmse = training(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_mae, val_rmse, val_r2, val_r, val_p, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
            "val_r": val_r,
            "val_p": val_p
        })

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"Train Loss: {train_loss:.3f} | "
            f"Train MAE: {train_mae:.3f} | "
            f"Train RMSE: {train_rmse:.3f} | "
            f"Val Loss: {val_loss:.3f} | "
            f"Val MAE: {val_mae:.3f} | "
            f"Val RMSE: {val_rmse:.3f} | "
            f"Val R2: {val_r2:.3f} | "
            f"Val r: {val_r:.3f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = float(val_mae)
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_mae": best_val_mae,
                "epoch": epoch
            }, save_path)

            print("Saved best model:", save_path)
            

if __name__ == "__main__":
    main()
