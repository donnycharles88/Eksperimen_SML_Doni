import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy import stats

def preprocess_computer_data(input_path="computer_prices_all.csv", output_path="computer_prices_clean.csv"):
    """
    Melakukan preprocessing dataset komputer secara otomatis
    dan mengembalikan DataFrame bersih yang siap dilatih.
    """
    # 1️⃣ Load data
    df = pd.read_csv(input_path)
    print("✅ Dataset dimuat:", df.shape)

    # 2️⃣ Menghapus missing values
    df = df.dropna()

    # 3️⃣ Menghapus data duplikat
    df = df.drop_duplicates()

    # 4️⃣ Normalisasi fitur numerik
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # 5️⃣ Deteksi dan hapus outlier (Z-Score)
    z = np.abs(stats.zscore(df[num_cols]))
    df = df[(z < 3).all(axis=1)]

    # 6️⃣ Encoding kolom kategorikal
    cat_cols = df.select_dtypes(include=['object']).columns
    encoder = LabelEncoder()
    for col in cat_cols:
        df[col] = encoder.fit_transform(df[col])

    # 7️⃣ Binning pada kolom harga
    df['price_bin'] = pd.qcut(df['price'], q=3, labels=['Low', 'Medium', 'High'])

    # 8️⃣ Simpan hasil ke file baru
    df.to_csv(output_path, index=False)
    print("✅ Dataset bersih tersimpan:", output_path)

    return df

# Jalankan otomatis jika file dieksekusi langsung
if __name__ == "__main__":
    preprocess_computer_data()
