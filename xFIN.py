import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# =========================
# 1. Data
# =========================
data = [
    ("Develop software applications", "62.01"),
    ("Build mobile apps", "62.01"),
    ("Hospital patient care", "86.10"),
    ("Medical clinic services", "86.10"),
    ("Teach children in school", "85.20"),
    ("Primary education teacher", "85.20"),
]

df = pd.DataFrame(data, columns=["text", "label"])

print("\nDataset:")
print(df)

# =========================
# 2. Embedding model
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed all texts
embeddings = model.encode(df["text"].tolist())

# =========================
# 3. Klassificering via similarity
# =========================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def predict(text):
    query = model.encode([text])[0]

    scores = []
    for emb in embeddings:
        scores.append(cosine_similarity(query, emb))

    idx = np.argmax(scores)
    return df.iloc[idx]["label"]


# =========================
# 4. Test
# =========================
test_text = "real estate portfolio management"

print("\nPrediction:")
print("Input:", test_text)
print("Predicted NACE:", predict(test_text))
