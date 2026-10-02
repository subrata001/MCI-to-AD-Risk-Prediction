# -------------------------------------------------------------------- #
# CSV harmonization, ADNI & OASIS split and training-fitted scaling/PCA
# -------------------------------------------------------------------- #

from datetime import datetime
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def prepare_common_columns(df):
    df = df.copy()

    # age_for_cox
    if "age_for_cox" not in df.columns:
        if "true_age" in df.columns:
            df["age_for_cox"] = df["true_age"]
        elif "AGE" in df.columns:
            df["age_for_cox"] = df["AGE"]
        elif "age" in df.columns:
            df["age_for_cox"] = df["age"]
        else:
            raise ValueError("No age column found. Need true_age, AGE, or age.")

    # sex_male
    if "sex_male" not in df.columns:
        if "GENDER" in df.columns:
            df["sex_male"] = df["GENDER"].astype(str).map({
                "1": 1, "2": 0,
                "Male": 1, "Female": 0,
                "M": 1, "F": 0,
                "male": 1, "female": 0
            })
        elif "PTGENDER" in df.columns:
            df["sex_male"] = df["PTGENDER"].astype(str).map({
                "Male": 1, "Female": 0,
                "M": 1, "F": 0,
                "1": 1, "2": 0,
                "male": 1, "female": 0
            })
        else:
            raise ValueError("No sex column found. Need sex_male, GENDER, or PTGENDER.")

    # EDUC column
    if "EDUC" not in df.columns:
        if "PTEDUCAT" in df.columns:
            df["EDUC"] = df["PTEDUCAT"]
        elif "education" in df.columns:
            df["EDUC"] = df["education"]
        else:
            raise ValueError("No education column found. Need EDUC, PTEDUCAT, or education.")

    # CDRSUM column or CDR-SB
    if "CDRSUM" not in df.columns:
        if "CDRSB" in df.columns:
            df["CDRSUM"] = df["CDRSB"]
        elif "cdrsum" in df.columns:
            df["CDRSUM"] = df["cdrsum"]
        else:
            raise ValueError("No CDRSUM/CDRSB column found.")

    # time_to_event_years
    if "time_to_event_years" not in df.columns:
        if "time_to_event_days" in df.columns:
            df["time_to_event_years"] = df["time_to_event_days"] / 365.25
        else:
            raise ValueError("No time_to_event_years or time_to_event_days column found.")

    # event
    if "event" not in df.columns:
        if "progression_group" in df.columns:
            df["event"] = df["progression_group"].map({
                "pMCI_to_AD": 1,
                "sMCI": 0
            })
        else:
            raise ValueError("No event column found.")

    return df


def load_data(adni_csv, oasis_csv, n_pca_components=20):
    adni_df = pd.read_csv(adni_csv)
    oasis_df = pd.read_csv(oasis_csv)
    adni_df["dataset"] = "ADNI"
    oasis_df["dataset"] = "OASIS3"
    adni_df = prepare_common_columns(adni_df)
    oasis_df = prepare_common_columns(oasis_df)

    # print("ADNI event counts:")
    # print(adni_df["event"].value_counts(dropna=False))

    # print("\nOASIS3 event counts:")
    # print(oasis_df["event"].value_counts(dropna=False))


    adni_df = adni_df[adni_df["time_to_event_years"] > 0].copy()
    oasis_df = oasis_df[oasis_df["time_to_event_years"] > 0].copy()

    # print("ADNI event counts:")
    # print(adni_df["event"].value_counts(dropna=False))

    # print("\nOASIS3 event counts:")
    # print(oasis_df["event"].value_counts(dropna=False))


    # Feature columns (Merge Clincal and MRI_Embeddings)

    clinical_cols = [
        "age_for_cox",
        "sex_male",
        "EDUC",
        "MMSE",
        "CDRSUM",
        "BrainAGE_corrected" #brain_age_gap (BAG)
    ]

    embedding_cols = [c for c in adni_df.columns if c.startswith("emb_")]

    print("Number of embedding columns:", len(embedding_cols))
    print("First embedding columns:", embedding_cols[:10])

    feature_cols = clinical_cols + embedding_cols

    duration_col = "time_to_event_years"
    event_col = "event"

    needed_cols = feature_cols + [duration_col, event_col] 

    adni_model_df = adni_df[needed_cols].dropna().copy()
    oasis_model_df = oasis_df[needed_cols].dropna().copy()

    adni_model_df[event_col] = adni_model_df[event_col].astype(int)
    oasis_model_df[event_col] = oasis_model_df[event_col].astype(int)

    adni_model_df[duration_col] = adni_model_df[duration_col].astype(float)
    oasis_model_df[duration_col] = oasis_model_df[duration_col].astype(float)

    # print("ADNI model df:", adni_model_df.shape)
    # print("OASIS3 model df:", oasis_model_df.shape)

    # print("\nADNI events:", adni_model_df[event_col].sum())
    # print("ADNI censored:", (adni_model_df[event_col] == 0).sum())

    # print("\nOASIS3 events:", oasis_model_df[event_col].sum())
    # print("OASIS3 censored:", (oasis_model_df[event_col] == 0).sum())


    ## Split ADNI into train/internal test

    train_df, internal_test_df = train_test_split(
        adni_model_df,
        test_size=0.2,
        random_state=42,
        stratify=adni_model_df[event_col]
    )

    external_test_df = oasis_model_df.copy()

    print("ADNI train:", train_df.shape)
    print("ADNI internal test:", internal_test_df.shape)
    print("OASIS3 external test:", external_test_df.shape)

    print("\nADNI train events:", train_df[event_col].sum())
    print("ADNI internal test events:", internal_test_df[event_col].sum())
    print("OASIS3 external test events:", external_test_df[event_col].sum())


    # Clinical scaling + MRI embedding PCA

    # Choose PCA components
    # n_pca_components is passed by training.py

    clinical_cols = [
        "age_for_cox",
        "sex_male",
        "EDUC",
        "MMSE",
        "CDRSUM",
        "BrainAGE_corrected"
    ]

    # MRI embedding columns
    embedding_cols = [c for c in train_df.columns if c.startswith("emb_")]

    print("Number of original MRI embedding columns:", len(embedding_cols))
    print("Using PCA components:", n_pca_components)

    # Separate clinical and MRI embedding features

    X_train_clinical_raw = train_df[clinical_cols].values.astype("float32")
    X_internal_clinical_raw = internal_test_df[clinical_cols].values.astype("float32")
    X_external_clinical_raw = external_test_df[clinical_cols].values.astype("float32")

    X_train_emb_raw = train_df[embedding_cols].values.astype("float32")
    X_internal_emb_raw = internal_test_df[embedding_cols].values.astype("float32")
    X_external_emb_raw = external_test_df[embedding_cols].values.astype("float32")

    # Scale clinical features
    # Fit only on ADNI training set

    clinical_scaler = StandardScaler()

    X_train_clinical = clinical_scaler.fit_transform(X_train_clinical_raw).astype("float32")
    X_internal_clinical = clinical_scaler.transform(X_internal_clinical_raw).astype("float32")
    X_external_clinical = clinical_scaler.transform(X_external_clinical_raw).astype("float32")

    # Scale MRI embeddings before PCA
    # Fit only on ADNI training set

    embedding_scaler = StandardScaler()

    X_train_emb_scaled = embedding_scaler.fit_transform(X_train_emb_raw).astype("float32")
    X_internal_emb_scaled = embedding_scaler.transform(X_internal_emb_raw).astype("float32")
    X_external_emb_scaled = embedding_scaler.transform(X_external_emb_raw).astype("float32")

    # PCA for MRI embeddings
    # Fit only on ADNI training set

    pca = PCA(n_components=n_pca_components, random_state=42)

    X_train_emb_pca = pca.fit_transform(X_train_emb_scaled).astype("float32")
    X_internal_emb_pca = pca.transform(X_internal_emb_scaled).astype("float32")
    X_external_emb_pca = pca.transform(X_external_emb_scaled).astype("float32")

    print("PCA explained variance ratio:")
    print(pca.explained_variance_ratio_)

    print("Total explained variance:")
    print(np.sum(pca.explained_variance_ratio_))


    # Combine clinical + PCA MRI features

    X_train = np.concatenate([X_train_clinical, X_train_emb_pca], axis=1).astype("float32")
    X_internal = np.concatenate([X_internal_clinical, X_internal_emb_pca], axis=1).astype("float32")
    X_external = np.concatenate([X_external_clinical, X_external_emb_pca], axis=1).astype("float32")

    # Survival labels

    duration_col = "time_to_event_years"
    event_col = "event"

    durations_train = train_df[duration_col].values.astype("float32")
    events_train = train_df[event_col].values.astype("float32")

    durations_internal = internal_test_df[duration_col].values.astype("float32")
    events_internal = internal_test_df[event_col].values.astype("float32")

    durations_external = external_test_df[duration_col].values.astype("float32")
    events_external = external_test_df[event_col].values.astype("float32")

    y_train = (durations_train, events_train)
    y_internal = (durations_internal, events_internal)
    y_external = (durations_external, events_external)

    # Final PCA feature names

    pca_cols = [f"emb_pca_{i+1}" for i in range(n_pca_components)]
    feature_cols = clinical_cols + pca_cols

    # print("Final train X:", X_train.shape)
    # print("Final internal X:", X_internal.shape)
    # print("Final external X:", X_external.shape)
    # print("Final number of features:", len(feature_cols))
    # print("Feature columns:", feature_cols)

    ## Save preprocessing bundle for internal and external test

    preprocessing_bundle = {
        # Fitted sklearn objects
        "clinical_scaler": clinical_scaler,
        "embedding_scaler": embedding_scaler,
        "pca": pca,

        # Exact input feature orders
        "clinical_cols": list(clinical_cols),
        "embedding_cols": list(embedding_cols),
        "pca_cols": list(pca_cols),
        "final_feature_cols": list(feature_cols),

        # Target information
        "duration_col": duration_col,
        "event_col": event_col,

        # PCA information
        "n_pca_components": int(n_pca_components),
        "pca_explained_variance_ratio": (
            pca.explained_variance_ratio_.copy()
        ),
        "pca_total_explained_variance": float(
            np.sum(pca.explained_variance_ratio_)
        ),

        # Data split information
        "train_indices": train_df.index.to_numpy(),
        "internal_test_indices": internal_test_df.index.to_numpy(),
        "external_test_indices": external_test_df.index.to_numpy(),

        # Reproducibility information
        "train_test_random_state": 42,
        "internal_test_fraction": 0.20,
        "adni_csv": str(adni_csv),
        "oasis_csv": str(oasis_csv),
        "sklearn_version": sklearn.__version__,
        "saved_at": datetime.now().isoformat(),

        # Description of the final DeepSurv input
        "final_input_order_description": (
            "Scaled clinical features followed by PCA-transformed "
            "scaled MRI embeddings"
        )
    }

    return dict(X_train=X_train, X_internal=X_internal, X_external=X_external, y_train=y_train, y_internal=y_internal, y_external=y_external, preprocessing_bundle=preprocessing_bundle)
