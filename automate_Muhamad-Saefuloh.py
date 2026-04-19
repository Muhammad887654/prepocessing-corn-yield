"""
automate_Muhamad-Saefuloh.py
============================
Script otomatis preprocessing dataset jagung.
Dijalankan via GitHub Actions setiap ada perubahan dataset.
Output: preprocessed_corn_data.csv
"""

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess_data(df_raw: pd.DataFrame) -> tuple:
    df = df_raw.copy()

    # 1. Handle Missing Values
    df['Acreage'].fillna(df['Acreage'].median(), inplace=True)
    df['Education'].fillna(df['Education'].mode()[0], inplace=True)

    # 2. Hapus Data Duplikat
    df = df.drop_duplicates()

    # 3. One-Hot Encoding
    exclude_encoding = ['County', 'Farmer', 'Crop', 'Power source',
                        'Water source', 'Crop insurance']
    categorical_cols = [c for c in df.select_dtypes(include='object').columns
                        if c not in exclude_encoding]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    # 4. Min-Max Normalization
    num_cols = ['Household size', 'Acreage', 'Fertilizer amount',
                'Laborers', 'Yield', 'Latitude', 'Longitude']
    num_cols = [c for c in num_cols if c in df.columns]
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # 5. Outlier Handling (IQR Capping)
    for col in num_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

    # 6. Fertilizer binning
    if 'Fertilizer amount' in df.columns:
        df['Fertilizer_Category'] = pd.cut(
            df['Fertilizer amount'],
            bins=[-0.001, 0.33, 0.66, 1.001],
            labels=['Low', 'Medium', 'High']
        )

    # 7. Pisahkan X dan y
    drop_x = ['Yield', 'County', 'Farmer', 'Crop',
               'Power source', 'Water source', 'Crop insurance']
    X = df.drop(columns=[c for c in drop_x if c in df.columns])
    y = df['Yield'] if 'Yield' in df.columns else None

    return X, y, df


if __name__ == '__main__':
    # Cari file raw data
    raw_paths = ['dataset/corndata_raw', 'corndata_raw', 'corn_data.csv']
    raw_file  = next((p for p in raw_paths if os.path.exists(p)), None)

    if raw_file is None:
        print("ERROR: File raw data tidak ditemukan. Pastikan ada di folder dataset/")
        exit(1)

    print(f"Memuat raw data dari: {raw_file}")
    df_raw = pd.read_csv(raw_file)
    print(f"Raw data shape: {df_raw.shape}")

    X, y, df_full = preprocess_data(df_raw)

    output_path = 'preprocessed_corn_data.csv'
    df_full.to_csv(output_path, index=False)

    print(f"Preprocessing selesai!")
    print(f"Output: {output_path}")
    print(f"Shape : {df_full.shape}")
    print(f"Kolom : {df_full.columns.tolist()}")
