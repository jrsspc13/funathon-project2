# %%
import logging
import random

import mlflow
import polars as pl
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import WordPieceTokenizer
from torchTextClassifiers.value_encoder import ValueEncoder, DictEncoder

logger = logging.getLogger(__name__)
load_dotenv(override=True)

# %%
# =========================
# 1. LOAD DATA
# =========================
df = pl.read_parquet(
    "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
)

df = df.to_pandas()

print(df.head())
print(f"Total rows: {len(df)}")

# %%
# =========================
# 2. BASIC INFO
# =========================
n_classes = df["code"].nunique()
print("Number of classes:", n_classes)

# %%
# =========================
# 3. SPLIT DATA
# =========================
train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=42)
val_df, test_df = train_test_split(tmp_df, test_size=0.50, random_state=42)

X_train, y_train = train_df["label"].values, train_df["code"].values
X_val, y_val = val_df["label"].values, val_df["code"].values
X_test, y_test = test_df["label"].values, test_df["code"].values

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# %%
# =========================
# 4. VALUE ENCODER (FIXED)
# =========================
classes = sorted(df["code"].unique())
label_to_id = {label: i for i, label in enumerate(classes)}

dict_encoder = DictEncoder(label_to_id)
value_encoder = ValueEncoder(label_encoder=dict_encoder)

# %%
# =========================
# 5. TOKENIZER
# =========================
tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=10)
tokenizer.train(X_train)

# ✅ OBS: INGEN vocab_size hack behövs i din version

print("Output tensor size:", tokenizer.tokenize(X_train[0]).input_ids.shape)

# %%
# =========================
# 6. MODEL CONFIG (FIXED)
# =========================
embedding_dim = 96

model_config = ModelConfig(
    embedding_dim=embedding_dim,
    num_classes=n_classes,
)

# ✅ KRAV i din version
model_config.categorical_vocabulary_sizes = []
model_config.n_heads_label_attention = None

# %%
# =========================
# 7. INIT MODEL
# =========================
ttc = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
    value_encoder=value_encoder,
)

# %%
# =========================
# 8. TRAIN
# =========================
mlflow.set_experiment("funathon-2026-project2")
mlflow.pytorch.autolog()

training_config = TrainingConfig(
    num_epochs=1,
    batch_size=128,
    lr=5e-4,
    patience_early_stopping=5,
)

with mlflow.start_run() as run:

    ttc.train(
        X_train,
        y_train,
        training_config=training_config,
        X_val=X_val,
        y_val=y_val,
        verbose=True,
    )

    mlflow.log_artifacts(
        training_config.save_path,
        artifact_path="model_artifacts",
    )

# %%
# =========================
# 9. LOAD MODEL
# =========================
local_dir = mlflow.artifacts.download_artifacts(
    f"runs:/{run.info.run_id}/model_artifacts"
)

ttc_loaded = torchTextClassifiers.load(local_dir)

# %%
# =========================
# 10. PREDICT
# =========================
random_indices = random.sample(range(len(X_test)), 3)
example_texts = X_test[random_indices]
example_true_codes = y_test[random_indices]

results = ttc_loaded.predict(example_texts, top_k=3)

for i, text in enumerate(example_texts):
    print("\nText:", text)
    print("True:", example_true_codes[i])
    print("Pred:", results["prediction"][i])

# %%
# =========================
# 11. EVALUATION
# =========================
results_test = ttc_loaded.predict(X_test, top_k=1)
preds = results_test["prediction"].squeeze(1)

accuracy = (preds == y_test).mean()
print(f"\nTest accuracy: {accuracy:.4f}")