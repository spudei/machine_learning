from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def get_input_cols(raw_df: pd.DataFrame) -> List[str]:
    """Повертає список колонок, які використовуються як ознаки."""
    return raw_df.drop(columns=["Exited", "id", "CustomerId", "Surname"]).columns.tolist()


def get_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Повертає списки числових і категоріальних колонок."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_cols, categorical_cols


def encode_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    categorical_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """Кодує категоріальні ознаки за допомогою OneHotEncoder."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(X_train[categorical_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    train_encoded = pd.DataFrame(
        encoder.transform(X_train[categorical_cols]),
        columns=encoded_cols,
        index=X_train.index
    )

    val_encoded = pd.DataFrame(
        encoder.transform(X_val[categorical_cols]),
        columns=encoded_cols,
        index=X_val.index
    )

    X_train = pd.concat([X_train.drop(columns=categorical_cols), train_encoded], axis=1)
    X_val = pd.concat([X_val.drop(columns=categorical_cols), val_encoded], axis=1)

    return X_train, X_val, encoder


def scale_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    numeric_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Масштабує числові ознаки за допомогою MinMaxScaler."""
    scaler = MinMaxScaler()
    scaler.fit(X_train[numeric_cols])

    X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])

    return X_train, X_val, scaler


def preprocess_data(
    raw_df: pd.DataFrame,
    scaler_numeric: bool = True
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    List[str],
    Optional[MinMaxScaler],
    OneHotEncoder
]:
    """Виконує попередню обробку train і validation даних."""
    input_cols = get_input_cols(raw_df)
    target_col = "Exited"

    train_df, val_df = train_test_split(
        raw_df,
        test_size=0.25,
        random_state=42,
        stratify=raw_df[target_col]
    )

    X_train = train_df[input_cols].copy()
    train_targets = train_df[target_col].copy()
    X_val = val_df[input_cols].copy()
    val_targets = val_df[target_col].copy()

    numeric_cols, categorical_cols = get_column_types(X_train)

    X_train, X_val, encoder = encode_data(X_train, X_val, categorical_cols)

    scaler = None
    if scaler_numeric:
        X_train, X_val, scaler = scale_data(X_train, X_val, numeric_cols)

    return X_train, train_targets, X_val, val_targets, input_cols, scaler, encoder


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    scaler: Optional[MinMaxScaler],
    encoder: OneHotEncoder,
    scaler_numeric: bool = True
) -> pd.DataFrame:
    """Обробляє нові дані за допомогою вже навчених scaler та encoder."""
    X = new_df[input_cols].copy()

    numeric_cols, categorical_cols = get_column_types(X)

    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(
        encoder.transform(X[categorical_cols]),
        columns=encoded_cols,
        index=X.index
    )

    X = pd.concat([X.drop(columns=categorical_cols), encoded_df], axis=1)

    if scaler_numeric and scaler is not None:
        X[numeric_cols] = scaler.transform(X[numeric_cols])

    return X