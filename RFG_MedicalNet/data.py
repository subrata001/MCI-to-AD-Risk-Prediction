import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

class BrainAgeDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Path to the preprocessed T1-weighted brain MRI volume in compressed NIfTI format (.nii.gz)
        img_path = row["preprocessed_image_path_hdbet"] 
        age = np.float32(row["age_label"])

        img = nib.load(img_path).get_fdata().astype(np.float32)

        # Replace NaN/inf if any
        img = np.nan_to_num(img)

        # Shape: (D, H, W) → add channel: (1, D, H, W)
        img = np.expand_dims(img, axis=0)

        img_tensor = torch.tensor(img, dtype=torch.float32)
        age_tensor = torch.tensor(age, dtype=torch.float32)

        return img_tensor, age_tensor


