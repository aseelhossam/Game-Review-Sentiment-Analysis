import os
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class GloVeTransformer:
    # (نفس الكلاس السابق بدون تغيير)
    def __init__(self, glove_path):
        self.embeddings = {}
        if not os.path.exists(glove_path):
            raise FileNotFoundError(f"GloVe file not found: {glove_path}")
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                self.embeddings[values[0]] = np.asarray(values[1:], dtype='float32')
        self.dim = len(next(iter(self.embeddings.values())))
        print(f"Loaded {len(self.embeddings)} word vectors (dim={self.dim}).")

    def _tokenize(self, text):
        return re.findall(r'\b\w+\b', str(text).lower())

    def transform_one(self, text):
        words = self._tokenize(text)
        vecs = [self.embeddings[w] for w in words if w in self.embeddings]
        if len(vecs) == 0:
            return np.zeros(self.dim)
        return np.mean(vecs, axis=0)

    def transform(self, texts):
        return np.array([self.transform_one(t) for t in texts])


def generate_glove_features_with_extra_columns(
    input_path,
    glove_model_or_path,
    output_train_path=None,
    output_test_path=None,
    transformer_save_path=None,
    test_size=0.2,
    random_state=42
):
    """
    Like generate_glove_features, but preserves ALL columns from the input CSV
    (e.g., 'game_name') in the output train/test files.
    """
    # 1. Load / create transformer
    if isinstance(glove_model_or_path, str):
        transformer = GloVeTransformer(glove_model_or_path)
    else:
        transformer = glove_model_or_path

    # 2. Load data
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = pd.read_csv(input_path)

    required = ["review_text", "final_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    df = df.dropna(subset=required).copy()
    df["review_text"] = df["review_text"].astype(str)

    # الأعمدة الإضافية التي نريد الاحتفاظ بها (كل ما عدا review_text و final_label)
    extra_cols = [col for col in df.columns if col not in required]

    # 3. Split (stratify by final_label)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["final_label"]
    )
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # 4. Generate GloVe vectors
    X_train = transformer.transform(train_df["review_text"])
    X_test = transformer.transform(test_df["review_text"])

    # 5. Build DataFrames
    dim = X_train.shape[1]
    train_vec_df = pd.DataFrame(X_train, columns=[f'glove_{i}' for i in range(dim)])
    test_vec_df = pd.DataFrame(X_test, columns=[f'glove_{i}' for i in range(dim)])

    # ضم الأعمدة الأساسية + الإضافية من البيانات الأصلية
    train_base = train_df[["review_text", "final_label"] + extra_cols].reset_index(drop=True)
    test_base = test_df[["review_text", "final_label"] + extra_cols].reset_index(drop=True)

    train_out = pd.concat([train_base, train_vec_df.reset_index(drop=True)], axis=1)
    test_out = pd.concat([test_base, test_vec_df.reset_index(drop=True)], axis=1)

    # 6. Save
    if output_train_path:
        train_out.to_csv(output_train_path, index=False)
        print(f"Train GloVe features saved: {output_train_path}")
    if output_test_path:
        test_out.to_csv(output_test_path, index=False)
        print(f"Test GloVe features saved: {output_test_path}")
    if transformer_save_path:
        with open(transformer_save_path, 'wb') as f:
            pickle.dump(transformer, f)
        print(f"Transformer saved: {transformer_save_path}")

    return train_out, test_out, transformer


# ── مثال للاستخدام مع الملفات الثلاث ──────────────────────────
if __name__ == "__main__":
    glove_path = "glove.6B.100d.txt"  # تأكد من المسار الصحيح
    if os.path.exists(glove_path):
        transformer = GloVeTransformer(glove_path)  # نحمّله مرة واحدة للكل
        schemes = [
            ("clean_v1.csv", "glove_train_s1.csv", "glove_test_s1.csv", "glove_transformer_s1.pkl"),
            ("clean_v2.csv", "glove_train_s2.csv", "glove_test_s2.csv", "glove_transformer_s2.pkl"),
            ("clean_v3.csv", "glove_train_s3.csv", "glove_test_s3.csv", "glove_transformer_s3.pkl"),
        ]
        for inp, train_path, test_path, trans_path in schemes:
            if os.path.exists(inp):
                # يمكنك استخدام نفس المحول لجميع الملفات لأن الجمل مبنية على نفس اللغة
                generate_glove_features_with_extra_columns(inp, transformer, train_path, test_path, trans_path)
            else:
                print(f"Warning: {inp} not found, skipping.")
    else:
        print("GloVe file not found. Please check path.")