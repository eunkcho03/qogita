import os, torch, pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"  # fast, 384-d

def build_text(title, brand):
    t = (str(title) or "").strip()
    b = (str(brand) or "").strip()
    return f"{t}. brand: {b}" if b else t



def get_or_make_embeddings(df, split, cache_dir="models/artifacts/text/text_cash"):
    os.makedirs(cache_dir, exist_ok=True)
    pt_path = os.path.join(cache_dir, f"text_emb_{split}.pt")
    if os.path.exists(pt_path):
        return torch.load(pt_path, map_location="cpu")

    model = SentenceTransformer(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")
    texts = [build_text(t, b) for t, b in zip(df["Name"], df["Brand"])]
    emb = model.encode(texts, batch_size=256, convert_to_tensor=True, show_progress_bar=True)
    emb = emb.cpu()
    torch.save(emb, pt_path)
    return emb

files = {
    "train": "train2.csv",
    "validation": "validation2.csv",
    "test": "test2.csv"
}

for split, filename in files.items():
    df = pd.read_csv(filename)
    get_or_make_embeddings(df, split)